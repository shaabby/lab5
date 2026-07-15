"""EuroSAT 数据管线；训练、验证和最终测试入口相互分离。"""

from pathlib import Path

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter


CLASS_NAMES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 64
NORMALIZE_MEAN = (127.5, 127.5, 127.5)
NORMALIZE_STD = (127.5, 127.5, 127.5)


def _check_split_directory(split_directory):
    split_directory = Path(split_directory).expanduser().resolve()
    if not split_directory.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{split_directory}")

    actual_classes = {path.name for path in split_directory.iterdir() if path.is_dir()}
    expected_classes = set(CLASS_NAMES)
    if actual_classes != expected_classes:
        missing = sorted(expected_classes - actual_classes)
        extra = sorted(actual_classes - expected_classes)
        raise ValueError(f"类别目录不匹配，缺少={missing}，多出={extra}")
    return split_directory


def _build_dataset(split_directory, batch_size, workers, training, num_samples=None):
    split_directory = _check_split_directory(split_directory)
    dataset = ds.ImageFolderDataset(
        str(split_directory),
        class_indexing=CLASS_TO_INDEX,
        extensions=[".jpg", ".jpeg", ".JPG", ".JPEG"],
        num_parallel_workers=workers,
        shuffle=training or num_samples is not None,
        decode=False,
    )
    if num_samples is not None:
        dataset = dataset.take(num_samples)

    if training:
        image_operations = [
            vision.Decode(),
            vision.RandomResizedCrop(
                (IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.70, 1.0),
                ratio=(0.80, 1.25),
                interpolation=Inter.BICUBIC,
            ),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomVerticalFlip(prob=0.5),
            vision.RandomColorAdjust(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.04,
            ),
            vision.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            vision.HWC2CHW(),
        ]
    else:
        image_operations = [
            vision.Decode(),
            vision.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Inter.BICUBIC),
            vision.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            vision.HWC2CHW(),
        ]

    dataset = dataset.map(
        operations=image_operations,
        input_columns="image",
        num_parallel_workers=workers,
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(ms.int32),
        input_columns="label",
        num_parallel_workers=workers,
    )
    return dataset.batch(batch_size, drop_remainder=training)


def build_training_dataset(data_root, batch_size, workers, num_samples=None):
    """仅加载 train/。"""
    return _build_dataset(
        Path(data_root) / "train",
        batch_size=batch_size,
        workers=workers,
        training=True,
        num_samples=num_samples,
    )


def build_validation_dataset(data_root, batch_size, workers, num_samples=None):
    """仅加载 val/，不用于参数更新。"""
    return _build_dataset(
        Path(data_root) / "val",
        batch_size=batch_size,
        workers=workers,
        training=False,
        num_samples=num_samples,
    )


def build_final_test_dataset(data_root, batch_size, workers):
    """仅供 evaluate_test_once.py 在最终评估时调用。"""
    return _build_dataset(
        Path(data_root) / "test",
        batch_size=batch_size,
        workers=workers,
        training=False,
    )
