"""只使用 train/ 更新参数、只使用 val/ 选模的训练入口。"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from mindspore import Tensor, nn
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor

from data import CLASS_NAMES, build_training_dataset, build_validation_dataset
from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="训练 GeoFusionNet")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device-id", type=int, default=int(os.getenv("DEVICE_ID", "0")))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    parser.add_argument("--amp-level", choices=("O0", "O2", "O3"), default="O2")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0 or args.epochs <= 0 or args.workers <= 0:
        raise ValueError("batch-size、epochs 和 workers 必须为正数")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs 必须满足 0 <= warmup-epochs < epochs")
    if args.patience <= 0 or args.max_minutes <= 0:
        raise ValueError("patience 和 max-minutes 必须为正数")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing 必须位于 [0, 1)")


def cosine_learning_rate(base_lr, min_lr, epochs, warmup_epochs, steps_per_epoch):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    values = np.empty(total_steps, dtype=np.float32)

    for step in range(total_steps):
        if step < warmup_steps:
            values[step] = base_lr * float(step + 1) / max(1, warmup_steps)
        else:
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
            values[step] = min_lr + 0.5 * (base_lr - min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
    return Tensor(values, ms.float32)


def optimizer_parameter_groups(network, weight_decay):
    decay_parameters = []
    no_decay_parameters = []
    all_parameters = network.trainable_params()

    for parameter in all_parameters:
        if len(parameter.shape) == 1 or parameter.name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
        {"order_params": all_parameters},
    ]


def evaluate_accuracy(network, dataset, loss_fn):
    """显式计算平均损失与 Top-1 accuracy。"""
    network.set_train(False)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataset.create_tuple_iterator(num_epochs=1):
        logits = network(images)
        loss = loss_fn(logits.astype(ms.float32), labels)
        logits_numpy = logits.asnumpy()
        labels_numpy = labels.asnumpy()
        batch_examples = int(labels_numpy.shape[0])

        total_loss += float(loss.asnumpy()) * batch_examples
        total_correct += int((logits_numpy.argmax(axis=1) == labels_numpy).sum())
        total_examples += batch_examples

    if total_examples == 0:
        raise RuntimeError("评估数据集为空")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
    }


def write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


class ValidationController(Callback):
    def __init__(
        self,
        network,
        validation_dataset,
        loss_fn,
        output_dir,
        patience,
        max_minutes,
        config,
    ):
        super().__init__()
        self.network = network
        self.validation_dataset = validation_dataset
        self.loss_fn = loss_fn
        self.output_dir = output_dir
        self.patience = patience
        self.max_seconds = max_minutes * 60.0
        self.config = config
        self.best_accuracy = -1.0
        self.best_epoch = 0
        self.stale_epochs = 0
        self.started_at = time.monotonic()
        self.history = []

    def on_train_epoch_end(self, run_context):
        callback_parameters = run_context.original_args()
        epoch = int(callback_parameters.cur_epoch_num)
        metrics = evaluate_accuracy(
            self.network,
            self.validation_dataset,
            self.loss_fn,
        )
        elapsed_seconds = time.monotonic() - self.started_at
        improved = metrics["accuracy"] > self.best_accuracy

        if improved:
            self.best_accuracy = metrics["accuracy"]
            self.best_epoch = epoch
            self.stale_epochs = 0
            ms.save_checkpoint(self.network, str(self.output_dir / "best.ckpt"))
        else:
            self.stale_epochs += 1

        record = {
            "epoch": epoch,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "elapsed_seconds": elapsed_seconds,
        }
        self.history.append(record)
        write_json(
            self.output_dir / "training_summary.json",
            {
                "config": self.config,
                "class_names": CLASS_NAMES,
                "best_epoch": self.best_epoch,
                "best_val_accuracy": self.best_accuracy,
                "history": self.history,
            },
        )
        print(
            f"epoch={epoch} val_loss={metrics['loss']:.5f} "
            f"val_acc={metrics['accuracy']:.4%} "
            f"best={self.best_accuracy:.4%}@{self.best_epoch} "
            f"elapsed={elapsed_seconds / 60.0:.1f}min",
            flush=True,
        )

        self.network.set_train(True)
        if self.stale_epochs >= self.patience:
            print(f"验证集 accuracy 连续 {self.patience} 个 epoch 未提升，停止训练。")
            run_context.request_stop()
        elif elapsed_seconds >= self.max_seconds:
            print("达到训练时间上限，停止训练。")
            run_context.request_stop()


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ms.set_seed(args.seed)
    ds.config.set_seed(args.seed)
    np.random.seed(args.seed)
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target="Ascend",
        device_id=args.device_id,
    )

    training_dataset = build_training_dataset(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    validation_dataset = build_validation_dataset(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    steps_per_epoch = training_dataset.get_dataset_size()
    if steps_per_epoch <= 0:
        raise RuntimeError("训练数据集为空或 batch-size 过大")

    network = create_model(num_classes=len(CLASS_NAMES))
    learning_rate = cosine_learning_rate(
        args.base_lr,
        args.min_lr,
        args.epochs,
        args.warmup_epochs,
        steps_per_epoch,
    )
    optimizer = nn.AdamWeightDecay(
        optimizer_parameter_groups(network, args.weight_decay),
        learning_rate=learning_rate,
        weight_decay=0.0,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    model = ms.Model(
        network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        amp_level=args.amp_level,
    )

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output_dir"] = str(args.output_dir)
    config["steps_per_epoch"] = steps_per_epoch
    config["parameter_count"] = int(
        sum(np.prod(parameter.shape) for parameter in network.trainable_params())
    )
    controller = ValidationController(
        network=network,
        validation_dataset=validation_dataset,
        loss_fn=loss_fn,
        output_dir=args.output_dir,
        patience=args.patience,
        max_minutes=args.max_minutes,
        config=config,
    )

    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    model.train(
        args.epochs,
        training_dataset,
        callbacks=[
            LossMonitor(per_print_times=max(1, steps_per_epoch // 5)),
            TimeMonitor(data_size=steps_per_epoch),
            controller,
        ],
        dataset_sink_mode=True,
    )
    print(
        f"训练结束：best_val_acc={controller.best_accuracy:.4%}, "
        f"best_epoch={controller.best_epoch}, "
        f"checkpoint={args.output_dir / 'best.ckpt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
