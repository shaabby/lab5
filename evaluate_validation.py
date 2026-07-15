"""加载指定模型的 checkpoint，并只在 val/ 上执行评估。"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mindspore as ms
from mindspore import nn

from data import CLASS_NAMES, build_validation_dataset
from train import MODEL_FACTORIES, evaluate_accuracy, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="评估验证集")
    parser.add_argument("--model", choices=tuple(MODEL_FACTORIES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=int(os.getenv("DEVICE_ID", "0")))
    parser.add_argument(
        "--device-target",
        choices=("Ascend", "CPU"),
        default="Ascend",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{args.checkpoint}")
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("batch-size 和 workers 必须为正数")

    context_options = dict(mode=ms.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context_options["device_id"] = args.device_id
    ms.set_context(**context_options)

    network = MODEL_FACTORIES[args.model](num_classes=len(CLASS_NAMES))
    checkpoint_parameters = ms.load_checkpoint(str(args.checkpoint))
    parameters_not_loaded, checkpoint_entries_not_loaded = ms.load_param_into_net(
        network,
        checkpoint_parameters,
        strict_load=True,
    )
    if parameters_not_loaded or checkpoint_entries_not_loaded:
        raise RuntimeError(
            "checkpoint 与网络结构不一致："
            f"网络未加载={parameters_not_loaded}，"
            f"checkpoint 未使用={checkpoint_entries_not_loaded}"
        )

    validation_dataset = build_validation_dataset(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    metrics = evaluate_accuracy(network, validation_dataset, nn.CrossEntropyLoss())
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_examples": metrics["examples"],
        "validation_loss": metrics["loss"],
        "validation_top1_accuracy": metrics["accuracy"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "validation_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
