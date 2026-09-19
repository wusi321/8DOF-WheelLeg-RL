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

### 全地形（Rough）1000 次迭代

想先跑一轮全地形实验，用：

```bash
cd ~/Wheelleg/8DOF-WheelLeg-RL/Wheelleg-RL
NUM_ENVS=2048 ITERS=1000 bash scripts/train.sh rough
```

`scripts/train.sh` 的阶段映射是 `flat` → `Wheelleg-Flat-v0`、`rough` → `Wheelleg-Rough-v0`、`recovery` → `Wheelleg-Recovery-v0`。默认 `NUM_ENVS=2048`、`ITERS=5`，所以上面两个变量都要显式给。

**1000 次迭代确实能看到全部地形。** `terrain_levels_obstacle_release` 的 `release_schedule` 比较的是 `env.common_step_counter`，而每个迭代推进 `num_steps_per_env=24` 步，所以配置里的 `200*24 / 500*24 / 700*24` 对应**第 200 / 500 / 700 次迭代**：

| 迭代 | 放开的地形 |
|---|---|
| 0 | `flat`、`random_rough`、`perlin_noise`、`sloped_terrain`、`pyramid_stairs` |
| 200 | `random_grid` |
| 500 | `pyramid_stairs_inv` |
| 700 | `rc_wall` |

所以 1000 次迭代会覆盖全部 8 种地形。用 `ITERS=20` 冒烟只能见到初始那几种。

Rough 的奖励集合和 Flat 不同：它移除 `track_lin_vel` / `track_ang_vel`，换成按轴拆分的 `track_lin_vel_x_exp`、`track_lin_vel_y_exp`、`track_ang_vel_z_exp`（各权重 1.0、`std=0.25`），并设置 `rel_lateral_envs=0.20`，即 20% 的环境专门下发横向命令，另有 `command_y_levels` 按横向跟踪表现自适应放宽命令范围。**横向能力在 Rough 里本来就是重点。**

`NUM_ENVS=2048` 是 Rough 的设计值（地形生成器为 `num_rows=10`、`num_cols=20`）。显存不足或想加快迭代可以降到 `1024`，但地形课程的统计会更嘈杂。

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

非交互服务器建议使用环境变量（下面的值就是当前实际在用的）：

```bash
export WANDB_PROJECT=mjlab
export WANDB_ENTITY=liucunfu2005-          # 注意结尾的连字符
```

`play` 的 `--wandb-run-path` 要用 `<entity>/<project>/<run_id>`，即 `liucunfu2005-/mjlab/<run_id>`。

如果暂时不需要上传：

```bash
export WANDB_MODE=disabled
```

提示：W&B 登录成功不代表训练一定成功；仍需观察本地日志、GPU 利用率、奖励、episode length、NaN 和地形成功率。不要把 API key 写入 Git、README 或 shell 脚本。

## 6. Play 回放

先列出训练产物（**不要凭记忆猜文件名**）：

```bash
find logs/rsl_rl -name 'model_*.pt' -type f | sort
```

**checkpoint 命名规则**：训练在**第 0 轮、每 `save_interval`（=100）轮、以及最后一轮**各存一次。迭代循环是 `0 .. max_iterations-1`，所以 **`ITERS=1000` 的最终 checkpoint 是 `model_999.pt`，不是 `model_1000.pt`**。

| `ITERS` | 实际文件名 |
|---|---|
| 1000 | `model_0.pt`、`model_100.pt` … `model_900.pt`、**`model_999.pt`** |
| 500 | `model_0.pt` … `model_400.pt`、**`model_499.pt`** |
| 6000 | `model_0.pt` … `model_5900.pt`、**`model_5999.pt`** |

加载本地 checkpoint：

```bash
uv run play Wheelleg-Flat-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_flat/<run_name>/model_5999.pt \
  --num-envs 4 --viewer viser
```

粗糙地形和恢复任务：

```bash
uv run play Wheelleg-Rough-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_rough/<run_name>/model_999.pt \
  --num-envs 4 --viewer viser
```

从 W&B 加载（**推荐不指定文件名，会自动取迭代数最高的那个**，因此不会因为命名猜错而失败）：

```bash
uv run play Wheelleg-Rough-v0 \
  --wandb-run-path liucunfu2005-/mjlab/<run_id> \
  --num-envs 4 --viewer viser
```

指定 W&B 里的某个 checkpoint（名字必须真实存在，否则会报错并列出可用文件）：

```bash
uv run play Wheelleg-Rough-v0 \
  --wandb-run-path liucunfu2005-/mjlab/<run_id> \
  --wandb-checkpoint-name model_999.pt \
  --num-envs 4 --viewer viser
```

实体名 `liucunfu2005-`（**结尾带连字符**）和项目名 `mjlab` 是当前实际使用的；早期文档里的 `wheelleg-rl` 是错的。启动日志会打印 `Loading checkpoint: <文件名> (run: <run_id>, ...)`，**以此确认拉到的 run 是否正确**。

W&B 回放前必须先登录。`--checkpoint-file` 与 `--wandb-run-path` 二选一；如果两个都不给，训练策略回放会报错。下载缓存位于 `logs/rsl_rl/wandb_checkpoints/<run_id>/`。

降低回放显存：

```bash
uv run play Wheelleg-Rough-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_rough/<run_name>/model_999.pt \
  --num-envs 1 --viewer viser
```

录制视频：

```bash
uv run play Wheelleg-Flat-v0 \
  --checkpoint-file logs/rsl_rl/wheelleg_flat/<run_name>/model_5999.pt \
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
| `no_wheel_support` | 奖励 −10 | **两个轮子都离地**超过 **0.25 s** | 只罚「完全失去支撑」；抬单腿迈步不罚（见下） |
| `wheeled_stance_locomotion` | 奖励 **+3** | 有移动指令 + 实际移动 + 机身达标 | 按接近站姿高度线性给分；支撑按 `0.5+0.5×着地比例` 计，迈步不被打折 |
| `lateral_step` | 奖励 **+1** | 横向指令 + 抬一腿 + **确有横向位移** + 另一轮仍支撑 | 付钱给「迈步」本身，而不是让轮子侧滑或原地抖轮 |
| `leg_symmetry` | 奖励 **−1** | 左右腿姿态不一致 | **仅在地面平整且非横向移动时生效**；两轮不等高时完全解除（见下） |
| `body_level` | 奖励 **−12** | 机身倾离世界竖直 | 横滚始终计费；前倾 0.30 rad 内免费；后仰 1.5 倍计费（见下） |
| `base_height_l2` | 奖励 −4 | 目标 **0.145 m**，高速时按速度下调 | 始终生效，低姿态轻微扣分、趴地扣分明显 |
| `standing_pose` | 奖励 −0.5 | 标准站姿 | 轻微约束腿型，不阻止抬腿 |
| `fallen_too_long` | **终止** | 倒地持续超过 **5 s** | 摔倒不再立即重置（见下） |
| `time_out` / `nan_detection` | **终止** | — | 超时与数值异常 |

奖励按 `dt` 缩放，所以权重可以直接读作**每秒**（50 步/s × 0.02 s = 1）。实测惩罚预算：

```
跪行、机身触地、双轮抬起、有移动指令：
  净空 0.05 m -> 爬行 -61.5  机身 -10.0  失去支撑 -10.0  合计  -81.5 /s
  净空 0.02 m -> 爬行 -84.6  机身 -10.0  失去支撑 -10.0  合计 -104.6 /s
正常轮式站姿移动（0.145 m、双轮着地）：
  track_lin_vel +1.7  wheeled_stance_locomotion +3.0  wheel_contact_bonus +0.2  合计 ≈ +5 /s
```

**两者相差约 20 倍**，跪行不再是可行策略。

### 机身必须竖直，而不是顺着坡面倾斜

`body_level` 用**线性**倾角代价而不是平方：平方在竖直附近几乎是平的，压不住沿坡滚动产生的稳定偏置。实测旧配置下 `upward` 只有 0.5 权重，机身平均倾斜约 **22°** 而代价差仅约 0.29/s（对比总奖励约 4.4/s，只有 7%），所以策略选择整个斜在坡上。

```
代价 = |roll| + 1.5 × max(pitch, 0) + max(-pitch - 0.30, 0)     单位 rad
              ↑ 横滚始终罚    ↑ 后仰 1.5 倍        ↑ 前倾 0.30 rad 内免费
```

- **横滚始终计费** → 不会顺着横向坡面躺着走。
- **前倾额度随速度开放**（静止时为 0，≥0.6 m/s 时最多 0.30 rad ≈17°）→ 静止和慢速时**必须完全水平**，只有高速时才允许自然前倾，这是你要求的行为。
- **后仰按 1.5 倍计费** → 大速度指令下向后翻（wheelie）被直接压制。

### 左右不等高时必须「收高腿、伸低腿」

当左右轮踩在不同高度（横向斜坡，或**跨台阶时两腿分别站在台阶的上下两个面**），要让机身 z 轴保持竖直，唯一的办法就是**收缩高位那一侧的腿、伸长低位那一侧的腿**——这在关节上表现为左右**大腿/膝角度不相等**。

而 `leg_symmetry` 罚的正是这个差值，所以强对称惩罚会把机身锁成「两腿伸出一样长、整体顺着坡面倾斜」。上一版权重是 −10，实测的代价约 0.56/s，而当时唯一的姿态奖励 `upward`（平方、权重 0.5）在 22° 倾斜与竖直之间只差约 0.29/s —— **倾斜比保持水平更划算，所以它必然选择斜着走。**

现在的处理是把「什么时候对称才算正确」明确出来：

```
对称代价 = 腿型差值 × 地面平整度 × (1 − 横向指令强度)
```

- **`wheel_height_difference`**：直接取两个轮体在世界系的 z 之差。它度量的是**地形高低差**而不是机身姿态——顺坡倾斜和保持水平时这个差值是一样的，所以它正好回答「现在两条腿是否必须不等长」。差值达到 `UNEVEN_HEIGHT_REFERENCE = 0.03 m` 时对称要求**完全解除**。
- **横向指令**：`|cmd_y| ≥ 0.25 m/s` 时解除，因为横向移动本来就靠单腿迈步（不对称）。
- **权重 −10 → −1**：平地之外的残余偏好也降到可以忽略。

同时新增的 `body_level`（−12、线性）提供**保持水平**的压力，与上面的解除配合：一个负责「必须平」，一个负责「允许不平等的腿」。两者方向一致，不再互相抵消。

`joint_mirror`（rough 原有、镜像大腿/膝对、权重 −0.05）测的是同一个量且没有地形门控，会与上述目标冲突，已删除。

`joint_mirror` 删除后，`Metrics/wheel_height_diff_m` 可以直接观察机器人是否真的踩在不等高处；`Episode_Reward/body_level` 应接近 0，说明机身确实保持竖直。

### 摔倒不再立即重置，改成「长时间起不来才重置」

`bad_orientation`（倾斜 > 1.0 rad 立即重置）已删除，换成 `fallen_too_long`：倾斜超过 0.9 rad **或**机身触地算「倒地」，连续倒地超过 **5 s** 才结束 episode，中途站起来就清零计时。

这样才可能学会起身：之前一倾斜就重置，等于在它真正摔倒前就把它救回来，倒地后的几秒它从来没有机会练习站起。代价是失败 episode 会多跑几秒、样本效率略降，这是学起身必须付的成本。

### 地形难度：起点必须"几乎平"，由课程自己爬上去

**障碍高度按难度插值，而难度 = `level/(num_rows-1)`：**

```
step_height = range[0] + difficulty × (range[1] - range[0])
```

默认 `difficulty_range=(0.0, 1.0)` 时 level 0 的 `step_height` 恰好是 **0** —— 楼梯地形在 level 0 就是平地（源码里有显式分支处理这个退化情形）。上一轮实测 `Curriculum/terrain_levels/mean = 0.5563`，对应难度 0.062、台阶约 **7 mm**，**策略整轮基本在平地上训练，从没学过抬轮过台阶。**

但把下界直接抬到 0.3 是**错的**：`max_init_terrain_level=5` 会让一半环境从**第 5 级**起步，难度 0.3–0.65，也就是**从第一个 episode 起就面对 3.6–7.8 cm 的障碍**——这台 3 cm 轮半径的机器人站都站不住，于是策略崩了（见下）。

现在的取值是**下界 0.02（约 2.4 mm，等同平地但不退化）+ `max_init_terrain_level = 1`**，让所有环境从最简两级开始，再由**已经修好的课程**逐步抬升难度。

**同时修掉了地形课程的降级条件。** `terrain_levels_vel_strict` 原本用「距出生点的净位移」：晋级要 >4 m，降级是 < 指令速度 × 20 s × 0.33。而本任务 `rel_heading_envs = 1.0`，**所有环境都在转向行走** —— 以 0.25 m/s 前进、0.3 rad/s 偏航的机器人一个 episode 走约 4.8 m 弧长，但**净位移只有约 1.6 m**，低于降级线，**每轮都被降级**，难度被永久压在最底层。现在改用**路径长度**（新增 `PathLength` 指标，逐步累加实际行程并排除重置瞬移）。

**这一项已经验证有效**：课程均值从 **0.5563 升到 2.113**，最高 9 级，8 种地形全部放开。

### 摔倒状态不再叠加所有惩罚

第一次改「5 s 才重置」时出了严重问题：倒地的机器人同时吃到 `body_level −7.2/s`、`low_posture_locomotion −5.7/s`、`stair_lateral_yaw_drift −2.6/s`、`base_contact_penalty −1.8/s`、`no_wheel_support −0.85/s`、`leg_joint_acc_l2 −0.85/s`……**合计约 −20/s，而正奖励只有约 +0.5/s**，episode 回报跑到 **−392**，value loss 冲到 **48.7**，策略熵变成 **−3.47**、动作标准差 0.38 → 0.17 —— **策略坍缩成"躺着不动"，再也探索不回来。**

现在 `upright_gate` 统一门控：一旦判定倒地（倾角 > 0.9 rad **或**机身触地），`low_posture_locomotion`、`no_wheel_support`、`base_contact_penalty` **全部归零** —— 这些项的职责是塑造**步态**，而倒地不是一个步态选择。倒地期间只剩高度误差（最多 −4/s）和倾角代价（上限 `MAX_TILT_COST`，最多 −6/s）提供"站起来"的动力，量级从 −20/s 降到约 −10/s。

`MAX_DOWN_TIME` 同时从 5 s 收到 **3 s**：起身动作只需 1–2 s，3 s 足够练习，但惩罚暴露时间减半。

### 高速时允许降低身位

`MIN_CLEARANCE` 和站姿目标都会随前进行驶速度下调最多 **0.015 m**（速度 ≥0.6 m/s 时到满，倒车同样适用）：

```
移动时限高  0.13  -> 0.115 m
站姿目标    0.145 -> 0.130 m
```

这是你要求的「高速可适当降低身位防止后仰」。**代价要说清楚：此时可用的最低净空比原来的 13 cm 低 1.5 cm**，是放宽了你更早提的硬指标，只是限定在高速前进这个场景。如果不想放宽，把 `standing.py` 里的 `CROUCH_CLEARANCE_DROP` 设为 0 即可，其余逻辑不变。

### 横向移动：只能迈步，不能滚

两个轮子的轴都沿 `Y`，滚动方向是 `X`。**所以这台机器人像差速小车一样，物理上无法靠滚动横向移动**——`lin_vel_y` 只能靠抬腿迈步实现，否则就是轮子侧滑。因此：

- `no_wheel_support` 用**两个轮子 air time 的较小值**，即只在「**两个轮子同时离地**」时才开始计时。抬单腿迈步时另一个轮子仍在支撑，**无论抬多久都不罚**。这正是跪行（双轮全抬 + 机身触地）与正常迈步的区别。
- `wheeled_stance_locomotion` 的支撑项用 `0.5 + 0.5×着地比例`：单轮支撑得 0.75，双轮得 1.0，所以迈步不被惩罚，但完全失去轮子支撑会掉到 0.5 再被 `no_wheel_support` 接管。
- `lateral_step` 专门奖励「横向指令下抬一腿」，权重 +1/s。速度跟踪奖励只能看结果，**无法区分「迈步横移」和「轮子侧滑」**，所以需要单独奖励动作本身。它现在还要求**确实存在横向速度**（`|v_y| ≥ 0.10 m/s`）：上一版只要求「抬腿 + 另一轮支撑」，而实测 `wheel_contact_fraction = 0.4929`、`wheel_air_time_s = 0.0169 s`（不到一个控制步），说明轮子本来就在约 30 Hz 地反复离地接触，于是**单步抖动就能领到奖励**，这正是 play 里看不到真正横向迈步的原因之一。
- `leg_symmetry` 乘以 `1 − 横向指令强度`：`|cmd_y| ≥ 0.25 m/s` 时要求完全放开。**横向移动本来就必须左右不对称**（一条腿抬、一条腿撑），而前进/后退/转向不需要，所以只在前者放开。

### 对称性的几何依据

模型里两个髋关节轴**都是 `+X`（没有镜像）**。绕 `+X` 旋转 `a` 会把腿末端推向 `+y`，所以**镜像对称要求 `left_hip == -right_hip`，判断量是和**。大腿和膝关节轴都是 `+Y`，绕 `Y` 旋转不涉及横向偏移，相同角度本身就是镜像对称，判断量是**差**。髋只给 0.5 权重，保留横向平衡使用外展的自由度。

**两足轮腿没有「对角步态」**——对角是四足概念，这里只有左右对称这一个有意义的选项。

### 两个被移除的奖励

- **`feet_air_time`**：给「脚离地 0.1–0.5 s」加分，是给足式机器人鼓励抬腿迈步用的。对轮式机器人它是反的：**它直接付钱让策略把轮子抬起来**，正好鼓励了跪行。所以不是调权重，而是移除，用 `no_wheel_support` 反向替代。
- **rough 的 `abduction_mirror`**：它取两髋角之**差**，而上面证明了镜像对称应该取**和**。所以它在惩罚正确的对称外展、反而奖励会侧倾的同向模式，与 `leg_symmetry` 直接冲突，删除。

新增观测：`Metrics/wheel_air_time_s`、`Metrics/wheel_support_time_s`、`Metrics/wheel_contact_fraction`、`Metrics/lateral_command`。

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

`play` 时**不要加 `--no-terminations`**：它会把 `fallen_too_long` 也关掉，看到的行为就不等于训练行为了。注意摔倒后**不会立即重置**，会先在地上躺最多 5 s，这是正常的。

```bash
uv run python tests/test_stance.py        # 几何、阈值与配置契约，无需 MuJoCo/torch
uv run python tests/test_standing.py      # 数值约束，需要 torch/MuJoCo
```

### 与 16DOF 参考工程的对照

对照 `05_software-16DOF四轮足机械狗-MJlab-强化学习训练算法/train/rc_mjlab/`：

- **动作配置已经一致，无需改动**：腿部 `scale` 髋 0.125 / 其余 0.25、`use_default_offset=True`、低通 5 Hz；轮部 `scale=5.0`、低通 15 Hz；延迟 0–2 控制步（0–40 ms）；两者都没有显式动作裁剪。低通公式同为 `alpha = 1-exp(-2πf_c/f_s)`，且都是「先低通、后 scale」。
- 参考工程 rough **同样 pop 掉 `upright`**，只留 `upward`（权重 0.5）——它也存在姿态约束偏弱的问题，不是可以直接照搬的答案。
- 参考工程把「低姿态」做成**独立任务** `Robot-Crawl-v0`：初始高度 0.20 m、大腿 1.65、膝 −2.55，`bad_orientation` 放宽到 80°、**删除** `base_ground_contact` 终止，加单边高度奖励 `crawl_height_reward`（目标 0.22、权重 +1.5）和 `flat_orientation`（−1.0）。本项目没有另开爬行任务，而是把姿态从**终止条件**改成**奖励信号**，并保留单一 locomotion 任务。
- 参考工程**没有**任何「卡住/起不来」超时逻辑，也没有起身专用初始状态或奖励；`fallen_too_long` 是本项目新加的。
- 参考工程的地形课程阈值与改前完全相同（晋级 `size[0]/2`、降级系数 0.33、`difficulty_range` 默认 `(0,1)`），**所以它同样没解决 level 0 等于平地的问题**，这一项没有可照搬的先例。
- 参考工程 `joint_mirror` 的缓存实现有一个隐患：`env.joint_mirror_joints_cache` 在**首次调用**时按当次的 `mirror_joints` 建立，同一环境上挂两个不同参数的实例会复用第一份缓存。本项目没有沿用这个模式。

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
