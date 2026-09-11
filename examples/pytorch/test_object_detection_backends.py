#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    HfArgumentParser,
    ResNetConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import EvalPrediction


sys.path.append(str(Path(__file__).parent / "object-detection"))
import run_object_detection as trainer_example
import run_object_detection_no_trainer as accelerate_example


def require_ultrafast():
    pytest.importorskip("ultrafast_pycocotools")
    try:
        trainer_example.create_coco_metric("ultrafast")
    except ValueError:
        pytest.skip("requires the unreleased TorchMetrics ultrafast backend")


def test_backend_argument_parsers():
    parser = HfArgumentParser(trainer_example.DataTrainingArguments)
    assert parser.parse_args_into_dataclasses([])[0].coco_eval_backend == "pycocotools"
    assert parser.parse_args_into_dataclasses(["--coco_eval_backend", "ultrafast"])[0].coco_eval_backend == "ultrafast"
    with mock.patch.object(sys, "argv", ["example"]):
        assert accelerate_example.parse_args().coco_eval_backend == "pycocotools"
    with mock.patch.object(sys, "argv", ["example", "--coco_eval_backend", "ultrafast"]):
        assert accelerate_example.parse_args().coco_eval_backend == "ultrafast"
    with pytest.raises(SystemExit):
        parser.parse_args_into_dataclasses(["--coco_eval_backend", "unknown"])
    with mock.patch.object(sys, "argv", ["example", "--coco_eval_backend", "unknown"]), pytest.raises(SystemExit):
        accelerate_example.parse_args()


@pytest.mark.parametrize("example", [trainer_example, accelerate_example])
def test_unsupported_backend_has_actionable_error(example):
    with mock.patch.object(example, "MeanAveragePrecision", side_effect=ValueError("unsupported backend")):
        with pytest.raises(ValueError, match="TorchMetrics build with ultrafast support"):
            example.create_coco_metric("ultrafast")


def test_metric_callbacks_match_on_nontrivial_detections():
    require_ultrafast()
    processor = DetrImageProcessor()
    # Image 1 has a high-confidence false positive and lower-confidence match;
    # image 2 supplies a second class, and image 3 has no annotations.
    boxes = torch.tensor(
        [
            [[0.8, 0.8, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
            [[0.3, 0.3, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]],
            [[0.8, 0.8, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
        ]
    )
    logits = torch.tensor(
        [
            [[5.0, -3.0, -1.0], [3.0, -3.0, -1.0]],
            [[-3.0, 5.0, -1.0], [-3.0, -3.0, 5.0]],
            [[-3.0, -3.0, 5.0], [-3.0, -3.0, 5.0]],
        ]
    )
    targets = [
        {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]]) if i < 2 else torch.empty((0, 4)),
            "class_labels": torch.tensor([i]) if i < 2 else torch.empty(0, dtype=torch.long),
        }
        for i in range(3)
    ]
    evaluation = EvalPrediction(
        predictions=[(np.zeros(1), logits.numpy(), boxes.numpy())],
        label_ids=[[{k: v.numpy() for k, v in target.items()} for target in targets]],
    )
    metrics = [
        trainer_example.compute_metrics(
            evaluation, processor, id2label={0: "one", 1: "two"}, coco_eval_backend=backend
        )
        for backend in ("pycocotools", "ultrafast")
    ]
    assert metrics[0] == metrics[1]
    assert 0 < metrics[1]["map"] < 1

    class FixedModel(torch.nn.Module):
        def forward(self, **batch):
            return SimpleNamespace(logits=logits, pred_boxes=boxes)

    accelerator = Accelerator(cpu=True)
    for backend in ("pycocotools", "ultrafast"):
        actual = accelerate_example.evaluation_loop(
            FixedModel(),
            processor,
            accelerator,
            [{"labels": copy.deepcopy(targets)}],
            {0: "one", 1: "two"},
            coco_eval_backend=backend,
        )
        assert actual == metrics[0]


def tiny_detr_and_dataset():
    torch.manual_seed(42)
    backbone = ResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_features=["stage4"],
    )
    config = DetrConfig(
        backbone_config=backbone,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_queries=4,
        num_labels=2,
        dropout=0.0,
    )
    model = DetrForObjectDetection(config)
    processor = DetrImageProcessor(do_resize=False, do_rescale=False)
    dataset = []
    for image_id in range(4):
        annotations = {
            "image_id": image_id,
            "annotations": [
                {
                    "image_id": image_id,
                    "category_id": image_id % 2,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
        }
        sample = processor(images=torch.rand(3, 64, 64), annotations=annotations, return_tensors="pt")
        dataset.append({key: value[0] for key, value in sample.items()})
    return model, processor, dataset


def test_real_trainer_and_accelerate_evaluation(tmp_path):
    require_ultrafast()
    model, processor, dataset = tiny_detr_and_dataset()
    id2label = {0: "one", 1: "two"}
    outputs = []
    for backend in ("pycocotools", "ultrafast"):
        args = TrainingArguments(
            output_dir=str(tmp_path / backend),
            per_device_eval_batch_size=2,
            use_cpu=True,
            report_to="none",
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=dataset,
            data_collator=trainer_example.collate_fn,
            processing_class=processor,
            compute_metrics=partial(
                trainer_example.compute_metrics,
                image_processor=processor,
                id2label=id2label,
                coco_eval_backend=backend,
            ),
        )
        for _ in range(2):
            result = trainer.evaluate()
            outputs.append(
                {
                    key.removeprefix("eval_"): value
                    for key, value in result.items()
                    if key.startswith(("eval_map", "eval_mar"))
                }
            )
    assert all(result == outputs[0] for result in outputs)
    accelerator = Accelerator(cpu=True)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=accelerate_example.collate_fn)
    model, dataloader = accelerator.prepare(model, dataloader)
    for backend in ("pycocotools", "ultrafast"):
        for _ in range(2):
            result = accelerate_example.evaluation_loop(
                model, processor, accelerator, dataloader, id2label, coco_eval_backend=backend
            )
            assert result == outputs[0]
