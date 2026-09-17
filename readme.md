# 8DOF WheelLeg RL

这是 8DOF 双轮腿机器人的独立强化学习工程。训练框架采用 [mjlab](https://github.com/mujocolab/mjlab) + PPO，工程代码位于 [`Wheelleg-RL/`](Wheelleg-RL/)，机器人 URDF 与网格位于 [`robot_description/`](robot_description/)，CAD 源文件位于 [`mechanical/`](mechanical/)。原始 16DOF 机器狗和 MicroDuck 目录只作为参考，不属于本训练工程的运行时依赖。

## 设计目标

- 8 个执行关节：左右髋、左右大腿、左右膝、左右轮。
- 50 Hz 策略、200 Hz 物理仿真，actor 观测保留速度指令、姿态、关节位置/速度和上一动作。
- 基础课程：自动平衡站立、前后/横向移动、原地旋转、轮式与腿式运动。
- 进阶课程：平地移动、斜坡、布朗/Perlin 粗糙地形、矮墙和两种楼梯；障碍高度统一不超过 12 cm。
- 恢复课程：摔倒后站起、蹲下、趴下、翻滚。恢复奖励采用进度势函数，避免在坏状态上刷奖励。
- 观测默认盲训，可在环境配置中启用 height scanner/raycast critic；速度和关节速度设置上限，支持推扰、摩擦、质量、执行器和延迟随机化。
- W&B 记录项目名默认为 `wheelleg-rl`，服务器侧使用 uv 管理环境。

## 模型转换

canonical 输入是 `robot_description/urdf/8DOFROBOT2.urdf`。转换器会规范化 SolidWorks 导出的关节名空格，并生成训练使用的 `Wheelleg-RL/mjcf/8dof_wheelleg.xml`：

```bash
cd Wheelleg-RL
python scripts/urdf_to_mjcf.py ../robot_description/urdf/8DOFROBOT2.urdf mjcf/8dof_wheelleg.xml --mesh-dir ../robot_description/meshes
```

## Ubuntu 22.04 / GPU

```bash
cd Wheelleg-RL
bash scripts/setup_ubuntu.sh
uv run list-envs
```

先做 5 iteration smoke test，再启动长训练：

```bash
ITERS=5 NUM_ENVS=64 bash scripts/train.sh flat
ITERS=12000 NUM_ENVS=2048 bash scripts/train.sh flat
ITERS=12000 NUM_ENVS=2048 bash scripts/train.sh rough
ITERS=8000 NUM_ENVS=2048 bash scripts/train.sh recovery
```

Windows 本地可执行：

```powershell
cd Wheelleg-RL
.\scripts\train.ps1 -Stage flat -NumEnvs 64 -MaxIterations 5 -NoWandb
```

训练过程中建议固定记录：站立高度/姿态、各轴速度跟踪、轮速跟踪、动作变化率、关节速度越界率、摔倒率、恢复成功率、恢复耗时和每类地形成功率。ONNX 导出应沿用 mjlab 的导出入口并保留 observation normalizer。

## 目录边界

提交仓库时上传 `Wheelleg-RL/`、`robot_description/`、`mechanical/` 和本文件；不要上传 `05_software-16DOF四轮足机械狗-MJlab-强化学习训练算法/`、`microduck/`、`microduck_rl/` 和 `说明.txt`。

## 致谢与参考

本项目的训练结构和工程经验参考了 RC WheelLeg：<https://github.com/zeitvex/RC_WheelLeg>，以及 MicroDuck RL：<https://github.com/pollen-robotics/microduck_rl>。感谢相关作者和开源社区的工作。
