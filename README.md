# Lab5 参考实现：SimpleEuroSATCNN

该实现用于理解和二次修改。网络由四组 `Conv2d → ReLU → MaxPool2d` 和一个分类层组成，未调用高层经典模型封装。训练入口只构造 `train/` 与 `val/` 数据集；测试入口独立，并通过确认参数和持久化标记阻止重复评估。

## 1. 云端环境

目标镜像：`mindspore_2_3_ascend`，MindSpore 2.3.0、CANN 8.0.RC2、Python 3.9、Ascend 单卡。

镜像已经包含与 CANN 匹配的 MindSpore。创建虚拟环境时复用镜像包，无需重新安装 MindSpore：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import mindspore as ms; print(ms.__version__)"
```

输出应为 `2.3.0`。不要在该环境中安装 PyPI 的 CPU 版 MindSpore。

## 2. 数据目录

`--data-root` 指向包含以下三个目录的位置：

```text
eurosat_split/
├── train/
├── val/
└── test/
```

训练阶段不得运行测试脚本。

## 3. 训练与验证

首次运行使用单 epoch 冒烟训练，逐 step 输出进度并返回验证准确率：

```bash
./run_smoke.sh
```

冒烟配置使用 256 张训练图像、128 张验证图像、batch size 32、GRAPH 模式和 O0。数据下沉保持关闭，便于观察每个 step 的输出。

完整训练命令：

```bash
python train.py \
  --data-root /path/to/eurosat_split \
  --output-dir outputs \
  --device-id 0
```

默认配置为 batch size 128、最多 80 个 epoch、3 个 epoch 线性预热后余弦衰减、O2 混合精度、验证准确率 12 个 epoch 未提升时早停、总时间最多 120 分钟。每个 epoch 只在 `val/` 上计算一次准确率。最佳验证模型写入 `outputs/best.ckpt`，过程记录写入 `outputs/training_summary.json`。

首次云端运行先观察第一个 epoch。显存不足时只调整批大小，例如 `--batch-size 64`；数据加载报资源不足时只调整 `--workers 4`。改变 batch size 后建议按比例调整学习率：batch size 64 对应 `--base-lr 0.0015`。

### WideResNet8 实验

`WideResNet8` 使用一个 32 通道 stem，以及通道数为 32、64、128 的三个 BasicBlock。每个 BasicBlock 包含两层 3×3 卷积，降采样块使用 1×1 投影残差，最后连接全局平均池化和分类层。

只使用 `train/` 和 `val/` 运行该模型：

```bash
./run_resnet8.sh
```

模型和验证记录写入 `outputs_resnet8/`。训练成功结束后，脚本重新加载 `best.ckpt`，执行测试集评估并生成 `outputs_resnet8/test_result.json`。测试保护记录位于 `outputs_resnet8/TEST_EVALUATION_STARTED.json`。

## 4. 最终测试

确定当前运行即最终全量训练时，可使用一条命令在训练成功结束后自动执行唯一一次测试集评估：

```bash
./run_full_and_evaluate.sh
```

训练报错或被中断时，脚本不会读取测试集。正常完成、早停或达到训练时间上限后，脚本加载 `outputs/best.ckpt` 并执行测试。

如已单独完成全量训练，检查 `training_summary.json`，确认训练和所有超参数选择均已结束。随后只运行一次：

```bash
python evaluate_test_once.py \
  --data-root /path/to/eurosat_split \
  --checkpoint outputs/best.ckpt \
  --output-dir outputs \
  --device-id 0 \
  --confirm-final-test
```

脚本在读取 `test/` 前创建 `outputs/TEST_EVALUATION_STARTED.json`，完成后生成 `outputs/test_result.json`。任一记录已经存在时，脚本都会拒绝再次评估。

## 5. 文件职责

- `model.py`：自定义四层 SimpleEuroSATCNN；
- `model_resnet8.py`：独立的 WideResNet8 和 BasicBlock；
- `data.py`：分离的训练、验证和最终测试数据入口；
- `train.py`：训练、手写 accuracy 评估、验证选模、早停与限时；
- `evaluate_validation.py`：重新加载指定模型并只评估验证集；
- `evaluate_test_once.py`：加载最佳权重并执行唯一一次测试评估。
- `run_full_and_evaluate.sh`：全量训练成功结束后自动执行最终测试。
- `run_resnet8.sh`：训练 WideResNet8，并在成功结束后执行测试集评估。
