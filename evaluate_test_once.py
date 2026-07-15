"""在验证选模完成后，对 test/ 执行唯一一次最终评估。"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mindspore as ms
from mindspore import nn

from data import CLASS_NAMES, build_final_test_dataset
from train import MODEL_FACTORIES, evaluate_accuracy, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="执行唯一一次测试集评估")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=tuple(MODEL_FACTORIES), default="simple")
    parser.add_argument("--device-id", type=int, default=int(os.getenv("DEVICE_ID", "0")))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="确认验证选模已经结束，且现在执行唯一一次 test/ 评估",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.confirm_final_test:
        raise SystemExit("缺少 --confirm-final-test；未读取测试集。")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    guard_path = args.output_dir / "TEST_EVALUATION_STARTED.json"
    result_path = args.output_dir / "test_result.json"
    if guard_path.exists() or result_path.exists():
        raise SystemExit(
            "检测到测试评估已启动或已完成的记录，拒绝再次读取 test/。"
        )

    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target="Ascend",
        device_id=args.device_id,
    )
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

    guard = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
    }
    with guard_path.open("x", encoding="utf-8") as file:
        json.dump(guard, file, ensure_ascii=False, indent=2)
        file.write("\n")

    test_dataset = build_final_test_dataset(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    metrics = evaluate_accuracy(network, test_dataset, nn.CrossEntropyLoss())
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_examples": metrics["examples"],
        "test_loss": metrics["loss"],
        "test_top1_accuracy": metrics["accuracy"],
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
