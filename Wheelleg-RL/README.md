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

机器人关节为 8 个：`left_hip_joint`、`left_thigh_joint`、`left_knee_joint`、`left_wheel_joint`、`right_hip_joint`、`right_thigh_joint`、`right_knee_joint`、`right_wheel_joint`。参考站立角度来自 `robot_description/标准站立.txt`：髋 `0 rad`、大腿 `1.02 rad`、膝 `-1.57 rad`、轮 `0 rad`。该角度在平地落地时只有约 0.1255 m，低于 0.13 m 的要求，因此实际训练使用由此推导的等效站姿（髋 `0`、大腿 `0.8462`、膝 `-1.2333 rad`），详见[站姿与最低高度约束](#站姿与最低高度约束)。

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

## 站姿与最低高度约束

所有高度都是 `base_link` 坐标系原点距**正下方当地地面**的高度，不是机身底面的离地间隙，也不是地形绝对高度。用一条独立向下射线测量，射线未命中地面时视为“不可测”，不当作“过低”。

**高度不是终止条件。** 一低于阈值就重置，等于在机器人真正摔倒之前就把它救回来，它永远学不到该避免的后果。所以姿态和触地全部改成奖励信号：

| 机制 | 类型 | 阈值 | 作用 |
|---|---|---|---|
| `low_posture_locomotion` | 奖励 −100 | **移动中**低于 **0.13 m** | 线性惩罚，越低越重；停着不动时为 0 |
| `base_contact_penalty` | 奖励 −10 | 机身触地 | 趴在地上不重置，但持续扣分 |
| `wheel_off_ground` | 奖励 −10 | 单轮离地超过 **0.25 s** | 0.25 s 内免费，1.0 s 时饱和；轮子是轮式机器人的支撑，不是「抬脚」 |
| `wheeled_stance_locomotion` | 奖励 **+3** | 有移动指令 + 实际移动 + 机身达标 + 双轮着地 | 按接近站姿高度线性给分，跪行/抬轮/原地不动都拿不到 |
| `leg_symmetry` | 奖励 −10 | 左右腿姿态不一致 | 大腿/膝全权重，髋 0.5 权重（见下） |
| `base_height_l2` | 奖励 −4 | 目标 **0.145 m** | 始终生效，低姿态轻微扣分、趴地扣分明显 |
| `standing_pose` | 奖励 −0.5 | 标准站姿 | 轻微约束腿型，不阻止抬腿 |
| `bad_orientation` | **终止** | 倾斜超过 1.0 rad | 真正摔倒后才结束 episode |
| `time_out` / `nan_detection` | **终止** | — | 超时与数值异常 |

奖励按 `dt` 缩放，所以权重可以直接读作**每秒**（50 步/s × 0.02 s = 1）。实测惩罚预算：

```
跪行、机身触地、双轮抬起、有移动指令：
  净空 0.05 m -> 爬行 -61.5  机身 -10.0  抬轮 -10.0  合计  -81.5 /s
  净空 0.02 m -> 爬行 -84.6  机身 -10.0  抬轮 -10.0  合计 -104.6 /s
正常轮式站姿移动（0.145 m、双轮着地）：
  track_lin_vel +1.7  wheeled_stance_locomotion +3.0  wheel_contact_bonus +0.2  合计 ≈ +5 /s
```

**两者相差约 20 倍**，跪行不再是可行策略。

`leg_symmetry` 的度量方式有几何依据：模型里两个髋关节轴**都是 `+X`（没有镜像）**，所以左右腿视觉等高要求 `left_hip == -right_hip`，判断量是**和**；大腿和膝关节轴都是 `+Y`，绕 Y 旋转不涉及横向偏移，所以相同角度本身就是镜像对称，判断量是**差**。髋只给 0.5 权重，保留横向平衡和转向使用外展的自由度。**两足轮腿没有「对角步态」**——对角是四足概念，这里只有左右对称这一个有意义的选项。

**`feet_air_time` 已删除。** 这个奖励给「脚离地 0.1–0.5 s」加分，是给足式机器人鼓励抬腿迈步用的。对轮式机器人它是反的：**它直接付钱让策略把轮子抬起来**，正好鼓励了跪行。所以不是调权重，而是移除，并用 `wheel_off_ground` 反向的惩罚替代。同时新增 `Metrics/wheel_air_time_s` 和 `Metrics/wheel_contact_fraction` 作为替代观测。

行为含义：

- **停下来休息允许低位姿，甚至趴在地上**：只要不移动，爬行惩罚为 0，只有高度奖励的轻微扣分。
- **刚刷新时允许摔在地上**：惩罚只与「实际移动」挂钩，所以重置瞬间的落地过程不会被重置或重罚。
- **后面也允许摔倒**：摔倒本身不重置。只有真的倒过去（倾斜 > 1.0 rad）才结束 episode——那是摔倒的结果，不是被提前救回。
- **但移动时必须站起来**：`low_posture_locomotion` 的权重按每秒计入，0.12 m 移动约 −7.7/s，而速度跟踪奖励只有约 +1.4/s，趴着滑行严格亏本。惩罚是**线性**而不是平方：移动时 0.13 m 是要求而不是偏好，平方屏障在阈值下方几乎免费，会让策略稳定停在阈值下 1 cm 处爬行。
- 「移动」同时看平动速度和偏航角速度（`LINEAR_SPEED_MOVING = 0.15 m/s`、`YAW_RATE_MOVING = 0.3 rad/s`），所以原地蹲着旋转也算移动，不能绕过惩罚。

**为什么训练站姿不是 `标准站立.txt` 的角度。** 按当前模型几何，髋 `0`、大腿 `1.02`、膝 `-1.57 rad` 这个参考站姿在两轮落地时只能让 `base_link` 原点达到 **0.1255 m**，低于 0.13 m 的要求，所以它无法同时满足“用该角度”和“不低于 13 cm”。`src/wheelleg/stance.py` 从 MJCF 的连杆偏置推导几何，并求解出一个既满足高度、又把轮轴放在机身原点正下方的站姿（支撑线落在质心正下方，整机质心 x ≈ 0）：

```
参考站姿  hip 0.0000  thigh 1.0200  knee -1.5700  -> 0.1255 m，不满足 0.13
训练站姿  hip 0.0000  thigh 0.8462  knee -1.2333  -> 0.1450 m，余量 +1.5 cm
```

这个角度只是把腿稍微伸展（大腿约 48.5°、膝约 -70.7°），与参考站姿同族，并作为初始化、位置动作默认偏置和 `standing_pose` 姿态奖励目标；动作始终是相对站姿的偏移，不是锁死角度。几何或连杆改动后 `stance.py` 会重新求解，`tests/test_stance.py` 验证阈值顺序、余量、轮轴居中与质心位置。运行 `uv run python scripts/stance_kinematics.py` 可打印全部数值。

**已知取舍。** 0.13 m 是移动时的工作要求，而站姿目标是 0.145 m，两者相差 1.5 cm。上一轮 500 次训练的策略稳定在 0.1317 m，即刚好压着要求线：高度奖励在 0.1317 只有 −0.034/s，相对 1.4/s 的跟踪奖励几乎无效，而蹲低在物理上更省力（质心降低、倒立摆变短），所以策略理性地选择了「满足要求的最低姿态」。这是允许的（≥13 cm、两轮支撑、不趴行），但如果你希望移动时明显站得更高，需要把移动阈值的 `MIN_CLEARANCE` 抬到 0.138 或加大 `base_height_l2` 权重后**重新训练**。

Recovery 只放开姿态与触地终止（`enforce_standing=False`），不加爬行惩罚，否则刚落地就会被扣分。当前 Recovery 仍只是 Rough 的短 episode 配方，且仍保留了 `bad_orientation`，不能视为已实现完整恢复课程。

Rough 使用同一套约束。粗糙地形上“机身正下方的地面”会随台阶抬升，跨台阶时相对本地台阶的净空本来就会变小；当前楼梯/网格/矮墙高度上限 0.12 m 已接近机体站姿高度 0.145 m，因此 Rough 的爬行惩罚可能偏严。这是待验证项，不是已调好的结果，先以 Flat 验收。

**已移除正奖励裁剪。** 旧代码在 `_make_base_env_cfg` 里全局把每步总奖励裁剪到 `>= 0`，这会让爬行的惩罚只能把总分压到 0 而无法形成负反馈，与本节的惩罚目标直接冲突，且该补丁会泄漏到所有任务，因此连同 `mdp/only_positive_rewards.py` 一并删除，奖励语义恢复为标准形式。

机器人视觉网格与碰撞网格分离，只有 `*_collision` 几何参与接触，地形射线不扫描机器人自身。轮网格存在约 2.546 mm 的局部 Z 偏心，转换阶段修正其放置偏移，原始 URDF/STL 保留。初始根状态不再向地面以下随机（旧配置 z 方向 ±0.5 m 会把轮子压入地面）。

模型和奖励变更后须从头训练，不要沿用之前 checkpoint。

`play` 时**不要加 `--no-terminations`**：它会把 `bad_orientation` 也关掉，看到的行为就不等于训练行为了。

```bash
uv run python tests/test_stance.py        # 几何与阈值契约，无需 MuJoCo
uv run python tests/test_standing.py      # 数值约束，需要 torch/MuJoCo
```

## 执行器与力矩上限

这是**当前训练实际生效**的配置，唯一来源是 `src/wheelleg/actuator_spec.py`：

| 关节 | 仿真执行器 | 增益 | 力矩上限 |
|---|---|---|---|
| 6 个腿部关节（髋/大腿/膝） | `<position>` | `kp=35`、`kd=1.0` | **±4 N·m** |
| 2 个轮关节 | `<velocity>` | `kd=0.5` | **±2 N·m** |

力矩上限的作用方式是 `forcelimited=True` + `forcerange=±上限`，即无论位置误差多大，关节力矩都被裁到该范围。轮速指令本身**不裁剪**（`ctrllimited=False`），所以轮子没有显式最高转速，只有力矩上限在限制加速度。

**这些数字不是这台机器人的实测电机参数。** 原始 URDF 里每个关节都导出为 `<limit ... effort="0" velocity="0" />`，即**机器人描述文件完全没有力矩信息**。当前数值是从 16DOF 参考机（`05_software-16DOF...`，那台机器所有 16 个执行器都是 **17 N·m**，且机体高度 0.42 m）按比例手改下来的占位值。

按几何估算的**静态保持力矩**（整机 1.6 kg，单腿承担 7.8 N，`uv run python scripts/stance_kinematics.py` 可复现）：

```
hip    arm=0.0463 m  torque=0.363 N*m
thigh  arm=0.0312 m  torque=0.245 N*m
knee   arm=0.0362 m  torque=0.284 N*m
```

**站立只需约 0.36 N·m，而限制是 4 N·m，约 11 倍余量。** 所以：

- 力矩上限**不是**当前动作受限的原因，之前爬行也不是力矩不够造成的。
- 反过来，4 N·m 对这台 1.6 kg 的机器人偏宽松，仿真允许的加速/跳跃能力可能超过真机。若要 sim-to-real，需要把上限降到真机能给的值。

**要改成真机参数，需要你提供其中之一**：电机型号 + 减速比，或减速器输出端的额定/峰值力矩，或「电流上限 × 转矩常数 Kt」。拿到后只需改 `actuator_spec.py`、重新运行 `uv run python scripts/urdf_to_mjcf.py` 生成 XML，`tests/test_stance.py` 会校验 XML 与运行时配置没有脱节。

生成 MJCF 中的 `<actuator>` 块只是为了能用 MuJoCo 单独打开查看；训练时 `robot_cfg.get_spec()` 会删除该块并按本表重建，因此以本表为准。

## 参考与致谢

训练设计参考了 [RC WheelLeg](https://github.com/zeitvex/RC_WheelLeg)、[MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) 和 [MJLab](https://github.com/mujocolab/mjlab)。感谢相关作者和开源社区。
