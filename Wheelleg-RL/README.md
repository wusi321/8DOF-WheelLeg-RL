# Wheelleg-RL：8DOF 双轮腿强化学习

本目录是 8DOF 双轮腿机器人的独立训练工程，基于本地可编辑的 **MJLab + MuJoCo Warp + PPO**。这里只维护当前 8DOF 项目；16DOF 四足参考代码和 MicroDuck 参考代码位于仓库外部参考目录，不是本项目运行依赖。

## 1. 当前工程内容

```text
Wheelleg-RL/
├─ src/wheelleg/              # 8DOF 任务、机器人配置、奖励、课程
├─ mjcf/8dof_wheelleg.xml     # 由 8DOF URDF 转换的 MuJoCo 模型
├─ mjcf/meshes/               # 8DOF 左右腿和轮子网格
├─ mjlab/                     # 固定版本的本地 MJLab 依赖
├─ scripts/                   # 安装、转换、训练脚本
├─ configs/training.yaml      # 训练课程和 W&B 选项
├─ tests/                     # 模型契约测试
├─ pyproject.toml
└─ uv.lock
```

当前注册任务只有：

| 任务 | 用途 |
|---|---|
| `Wheelleg-Flat-v0` | 平地站立、平衡、前后/横向移动、原地旋转 |
| `Wheelleg-Rough-v0` | 斜坡、随机粗糙地形、矮墙和低矮楼梯 |
| `Wheelleg-Recovery-v0` | 摔倒恢复、站起和恢复动作训练 |

机器人关节为 8 个：`left_hip_joint`、`left_thigh_joint`、`left_knee_joint`、`left_wheel_joint`、`right_hip_joint`、`right_thigh_joint`、`right_knee_joint`、`right_wheel_joint`。标准站立角度来自 `robot_description/标准站立.txt`：髋 `0 rad`、大腿 `1.02 rad`、膝 `-1.57 rad`、轮 `0 rad`。

## 2. 服务器安装

要求：Ubuntu 22.04、Python 3.10–3.13、NVIDIA 驱动/CUDA、可用 GPU、`uv`。从任意目录执行均可，脚本会自动定位项目根目录：

```bash
bash /root/Wheelleg/8DOF-WheelLeg-RL/Wheelleg-RL/scripts/setup_ubuntu.sh
```

安装脚本默认测速并分别选择最快的 APT 与 PyPI 镜像。候选源包括清华、阿里云、中科大、腾讯云、华为云、北外、上海交大和南京大学。

```bash
# 交互选择
bash scripts/setup_ubuntu.sh --interactive

# 固定同一个源
WHEELLEG_MIRROR=ustc bash scripts/setup_ubuntu.sh

# APT 与 PyPI 分开指定
WHEELLEG_APT_MIRROR=aliyun \
WHEELLEG_PYPI_MIRROR=tuna \
bash scripts/setup_ubuntu.sh
```

脚本会将选中的 APT 源写入 `/etc/apt/sources.list.d/wheelleg-mirror.list`，不会删除系统原有源；本次 APT 操作只使用选中的源。Python 依赖通过当前进程的 `UV_INDEX_URL` 使用选中的 PyPI 源。若服务器禁止访问 `astral.sh`，请预先安装 `uv`，再运行脚本。

安装完成后检查：

```bash
cd /root/Wheelleg/8DOF-WheelLeg-RL/Wheelleg-RL
uv run list-envs | grep Wheelleg
nvidia-smi
```

## 3. 模型转换

输入模型是仓库根目录的 `robot_description/urdf/8DOFROBOT2.urdf`，网格来自 `robot_description/meshes`。重新转换：

```bash
bash scripts/convert_model.sh
```

转换器会修剪 SolidWorks 导出关节名的首尾空格，生成 `mjcf/8dof_wheelleg.xml`。转换后不要手动改关节顺序；训练配置依赖这些标准名称。

## 4. 训练

第一次必须先做小规模 smoke test：

```bash
NUM_ENVS=64 ITERS=5 bash scripts/train.sh flat
```

确认能正常启动后再训练：

```bash
NUM_ENVS=2048 ITERS=6000 bash scripts/train.sh flat
NUM_ENVS=2048 ITERS=12000 bash scripts/train.sh rough
NUM_ENVS=2048 ITERS=8000 bash scripts/train.sh recovery
```

也可以直接使用 MJLab CLI：

```bash
uv run train Wheelleg-Flat-v0 --env.scene.num-envs 2048 --agent.max_iterations 6000
```

训练日志默认位于 `logs/rsl_rl/<experiment_name>/`。不要在没有 GPU、CUDA 或正确 MJLab 环境的本地解释器中直接运行 `python`；本项目命令统一使用 `uv run`。

## 5. W&B 登录与记录

W&B 是可选但推荐的训练监控工具。服务器首次使用前：

```bash
uv run wandb login
# 或者
export WANDB_API_KEY="你的_W&B_API_KEY"
```

非交互服务器建议使用环境变量：

```bash
export WANDB_PROJECT=wheelleg-rl
export WANDB_ENTITY=你的账号或团队名
```

如果暂时不需要上传：

```bash
export WANDB_MODE=disabled
```

提示：W&B 登录成功不代表训练一定成功；仍需观察本地日志、GPU 利用率、奖励、episode length、NaN 和地形成功率。不要把 API key 写入 Git、README 或 shell 脚本。

## 6. Play 回放

当前支持 MJLab 的本地 checkpoint 和 W&B checkpoint 回放。先列出训练产物：

```bash
find logs/rsl_rl -name 'model_*.pt' -type f | sort
```

加载本地 checkpoint：

```bash
uv run play Wheelleg-Flat-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_flat/<run_name>/model_5000.pt
```

粗糙地形和恢复任务：

```bash
uv run play Wheelleg-Rough-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_rough/<run_name>/model_5000.pt

uv run play Wheelleg-Recovery-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_recovery/<run_name>/model_5000.pt
```

从 W&B 加载最新 checkpoint：

```bash
uv run play Wheelleg-Flat-v0 \
  --wandb-run-path <entity>/wheelleg-rl/<run_id>
```

从 W&B 加载指定 checkpoint：

```bash
uv run play Wheelleg-Rough-v0 \
  --wandb-run-path <entity>/wheelleg-rl/<run_id> \
  --wandb-checkpoint-name model_5000.pt
```

W&B 回放前必须先登录，并保证 run path、项目名和实体名正确。`--checkpoint-file` 与 `--wandb-run-path` 二选一；如果两个都不给，训练策略回放会报错。

降低回放显存：

```bash
uv run play Wheelleg-Rough-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_rough/<run_name>/model_5000.pt \
  --num-envs 1
```

录制视频：

```bash
uv run play Wheelleg-Flat-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_flat/<run_name>/model_5000.pt \
  --num-envs 1 --video --video-length 500
```

视频写入 checkpoint 对应实验目录下的 `videos/play/`。没有 checkpoint 时可以用零动作或随机动作检查环境初始化：

```bash
uv run play Wheelleg-Flat-v0 --agent zero --num-envs 4
uv run play Wheelleg-Flat-v0 --agent random --num-envs 4
```

## 7. 常见问题

- **`No pyproject.toml found`**：没有通过脚本定位项目根目录，使用仓库中的最新脚本，或先 `cd Wheelleg-RL`。
- **`Task not found`**：先运行 `uv run list-envs`，任务名必须是 `Wheelleg-*`，不是旧的 `Robot-*`。
- **W&B 找不到 run/checkpoint**：检查 `WANDB_API_KEY`、`<entity>/<project>/<run_id>` 和 checkpoint 文件名。
- **显存不足**：训练减小 `NUM_ENVS`，回放加 `--num-envs 1`。
- **CUDA 初始化失败**：检查 `nvidia-smi`、驱动版本、CUDA/PyTorch 安装和 `uv run python -c 'import torch; print(torch.cuda.is_available())'`。
- **模型关节找不到**：重新执行 `bash scripts/convert_model.sh`，检查 MJCF 关节名，不要恢复旧四足 XML。
- **回放窗口不显示**：使用支持图形界面的 SSH/X11，或启用服务器端虚拟显示；训练本身可无窗口运行。
- **脚本测速很慢**：使用 `WHEELLEG_MIRROR=ustc` 等环境变量跳过测速；每个候选源最多等待约 6 秒。
- **修改模型或环境后**：先执行 64 环境、5 iteration smoke test，再启动长训练。

## 8. 检查命令

```bash
uv run pytest tests/test_model_contract.py
uv run ruff check src scripts tests
uv run list-envs
```

本项目不包含真机驱动、ROS 2 部署和独立 ONNX Sim2Sim 工具；这些功能不属于当前训练工程的稳定接口。

## 参考与致谢

训练设计参考了 [RC WheelLeg](https://github.com/zeitvex/RC_WheelLeg)、[MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) 和 [MJLab](https://github.com/mujocolab/mjlab)。感谢相关作者和开源社区。
