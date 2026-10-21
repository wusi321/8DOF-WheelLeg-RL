# 8DOF WheelLeg RL

[![validate](https://github.com/wusi321/8DOF-WheelLeg-RL/actions/workflows/validate.yml/badge.svg)](https://github.com/wusi321/8DOF-WheelLeg-RL/actions/workflows/validate.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](Wheelleg-RL/pyproject.toml)

An open-source reinforcement-learning stack for an **8-DOF wheel-legged robot** — the
machine is a two-wheel balancing robot with a hip, thigh, knee and wheel per leg. The
repository covers the whole chain from robot description to a trainable policy:
URDF → MJCF conversion, an [mjlab](https://github.com/mujocolab/mjlab) + MuJoCo Warp
training pipeline (200 Hz physics, 50 Hz policy), PPO with a staged curriculum, reward
shaping, domain randomization, W&B logging, and a documented path from a checkpoint to
a deployable policy.

| | |
|---|---|
| **What** | 8 actuated joints (hip / thigh / knee / wheel per leg); three registered tasks: `Wheelleg-Flat-v0`, `Wheelleg-Rough-v0`, `Wheelleg-Recovery-v0` |
| **Stack** | Python 3.10+, mjlab + MuJoCo Warp, PyTorch, `uv` |
| **Status** | Active project. Development 2026-09-17 → 2026-10-09 produced 48 commits and 23 tagged training configurations; `v1.0.0` is the first documented release ([Releases](https://github.com/wusi321/8DOF-WheelLeg-RL/releases)) |
| **Maintainer** | [@wusi321](https://github.com/wusi321) — sole maintainer |
| **License** | MIT for this project's code; the vendored mjlab snapshot stays Apache-2.0 |
| **Related** | [RC_WheelLeg](https://github.com/zeitvex/RC_WheelLeg) — the 16-DOF competition platform this 8-DOF work precedes |

中文简介：这是 8DOF 双轮腿机器人的独立强化学习工程。训练框架采用 mjlab + PPO，工程代码位于
[`Wheelleg-RL/`](Wheelleg-RL/)；机器人 URDF、网格与参考站姿数据位于
[`robot_description/`](robot_description/)；机械 CAD 源文件位于 [`mechanical/`](mechanical/)。
设计取舍、奖励推导与实测数值记在 [`Wheelleg-RL/README.md`](Wheelleg-RL/README.md)
（中文深度文档，约 600 行，是判断这套工程是否真实的主要依据）。

## 仓库结构

```text
8DOF-WheelLeg-RL/
├─ Wheelleg-RL/
│  ├─ src/wheelleg/        # 任务注册、机器人配置、奖励、课程、站姿几何、地形（20 个模块）
│  ├─ mjcf/                # 由 URDF 转换的训练用 MJCF 与网格
│  ├─ scripts/             # 安装、URDF→MJCF 转换、站姿验算、训练启动器（sh/ps1）
│  ├─ configs/training.yaml# 课程、频率、W&B 配置
│  ├─ tests/               # 60 个无第三方依赖的契约测试 + 3 个需要 torch/MuJoCo 的测试
│  ├─ mjlab/               # 固定版本的 MJLab 快照（Apache-2.0，修改清单见 PROVENANCE.md）
│  └─ pyproject.toml, uv.lock
├─ robot_description/      # URDF、STL 网格、RViz/launch、参考站姿与收起姿态数据
└─ mechanical/             # SolidWorks 源文件、CNC 说明
```

## 快速开始

```bash
cd Wheelleg-RL
bash scripts/setup_ubuntu.sh          # Ubuntu 22.04 + CUDA，自动测速选择国内镜像
uv run list-envs | grep Wheelleg      # 应列出三个 Wheelleg-* 任务

# 任何长训练之前先跑 smoke test（约 95% 的配置错误能在这一步抓到）
ITERS=5 NUM_ENVS=64 bash scripts/train.sh flat

ITERS=12000 NUM_ENVS=2048 bash scripts/train.sh flat
ITERS=12000 NUM_ENVS=2048 bash scripts/train.sh rough
ITERS=8000  NUM_ENVS=2048 bash scripts/train.sh recovery
```

Windows 本地冒烟：

```powershell
cd Wheelleg-RL
.\scripts\train.ps1 -Stage flat -NumEnvs 64 -MaxIterations 5 -NoWandb
```

模型转换（canonical 输入是 `robot_description/8DOFROBOT2/urdf/8DOFROBOT2.urdf`）：

```bash
cd Wheelleg-RL
bash scripts/convert_model.sh
```

## 任务、课程与奖励

- **课程**：平地平衡 → 前后/横向移动与原地旋转 → 斜坡、Perlin/随机粗糙地形、矮墙、两种楼梯
  （障碍高度上限 12 cm）→ 摔倒起立与趴下/蹲下姿态。地形难度按课程等级插值，不是固定难度。
- **观测与动作**：盲训基线，可选 height scanner / raycast critic；`lin_vel_y` 只能靠抬腿迈步实现，
  因为两个轮轴都沿 Y、滚动方向是 X，物理上无法侧向滚过去。
- **奖励设计**：姿态与触地从终止条件改成奖励信号，倒地不再立即重置；恢复奖励用势能增量而不是
  「门控在坏状态上的正奖励」，避免策略趴着刷分。每一项权重、阈值与实测代价都在
  [`Wheelleg-RL/README.md`](Wheelleg-RL/README.md) 里有推导与数据。
- **执行器**：力矩上限（腿部 ±4 N·m、轮 ±2 N·m）是占位值，由 16DOF 参考机按比例折下来的，
  不是本机实测电机参数 —— 这一条在文档里写明，替换成真机参数只需改
  `src/wheelleg/actuator_spec.py` 后重跑转换器。

## 验证

不需要 GPU、不需要 MuJoCo、不需要 PyTorch 的契约测试（CI 每次都跑，共 60 个）：

```bash
cd Wheelleg-RL
python -m unittest -v tests.test_stance tests.test_urdf_geometry tests.test_entity_bindings
```

覆盖：站姿几何与高度约束、URDF→MJCF 转换的可复现性、轮网格同心性与碰撞体收敛、
MDP 项与 Wheelleg 实体的绑定。需要 MuJoCo/PyTorch 的测试（`test_model_contract.py`、
`test_standing.py`、`test_wheel_tracking.py`）在装好训练环境后运行。

## 范围边界

本仓库只承诺：8DOF 机器人 MJCF、mjlab 训练任务、PPO checkpoint 回放、W&B 训练监控、
基础视频录制。**真机驱动、ROS 2 部署与独立 ONNX Sim2Sim 工具不在本仓库范围内**，
这些工作属于 16DOF 平台，见 [RC_WheelLeg](https://github.com/zeitvex/RC_WheelLeg)。

## 参与贡献

Issue 与 Pull Request 都欢迎，见 [CONTRIBUTING.md](CONTRIBUTING.md)。提 bug 时请附上
`uv run list-envs` 输出、训练阶段（flat/rough/recovery）与失败前后的日志片段；
涉及奖励或课程的改动请附一次 64 环境 / 5 iteration 的 smoke test 结果。

## 第三方代码与许可

- [`Wheelleg-RL/mjlab/`](Wheelleg-RL/mjlab/) 是 **MJLab 的固定版本快照**（Apache-2.0），
  不是本项目原创代码：来源、修改清单与更新方式见
  [Wheelleg-RL/mjlab/PROVENANCE.md](Wheelleg-RL/mjlab/PROVENANCE.md)。
- 训练结构与工程经验参考了 RC_WheelLeg 与 MicroDuck RL，完整致谢与许可说明见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 引用

如果这套工程对你有用，引用信息见 [CITATION.cff](CITATION.cff)。

## 协议

本项目自有代码采用 [MIT License](LICENSE)。第三方依赖、模型、网格与参考项目仍以其各自的
许可证和版权声明为准。
