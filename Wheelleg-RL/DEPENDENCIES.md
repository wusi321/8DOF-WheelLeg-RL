# Wheelleg-RL 依赖说明

## 必需环境

- Ubuntu 22.04（服务器推荐）
- Python 3.10–3.13
- NVIDIA 驱动与 CUDA 12.8 兼容环境
- CUDA GPU（训练使用 MuJoCo Warp）
- `uv`
- `mjlab[cu128]`
- PyTorch CUDA
- `pynput`

项目使用本地可编辑 MJLab：

```toml
[tool.uv.sources]
mjlab = { path = "mjlab", editable = true }
```

精确依赖解析保存在 `uv.lock`。不要额外安装导航、ROS 2 或真机驱动依赖；这些内容不属于当前训练与回放工程。

## 安装

推荐使用：

```bash
bash scripts/setup_ubuntu.sh
```

该脚本支持国内 APT/PyPI 镜像测速、交互选择和环境变量固定源，详情见 `README.md`。

## 验证

```bash
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
uv run list-envs
uv run pytest tests/test_model_contract.py
```

若 `torch.cuda.is_available()` 为 `False`，不要直接启动长时间训练；先检查 `nvidia-smi`、驱动、CUDA 和 uv 解析到的 PyTorch wheel。

## MJLab 来源

本项目保留经过项目配置适配的本地 MJLab 工作树。其许可证位于 `mjlab/LICENSE`。上游项目：<https://github.com/mujocolab/mjlab>。

## 运行边界

当前工程只承诺：8DOF 机器人 MJCF、MJLab 训练任务、PPO checkpoint 回放、W&B 训练监控和基础视频录制。真机控制、ROS 2、独立 ONNX Sim2Sim、导航工具和历史四足模型不属于当前工程接口。
