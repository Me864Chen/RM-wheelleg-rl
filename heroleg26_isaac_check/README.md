# Isaac Gym 单机器人检查

全部操作使用普通用户环境。已有 `check_isaacgym.py` 只创建 actor、不步进物理；新 `check_pd.py` 执行真正的固定车体 PD 检查，不加载策略、不训练、不创建 Viewer。

## 固定车体 PD 检查

从当前仓库的 base → wheel_legged → heroleg26 配置读取站姿、PD、限幅和时间步。入口只解析所需的字面量赋值，不导入完整 RL 包；遇到不支持的配置表达式会报错，不猜测其含义。

训练环境与检查入口共用所选分支 `utils/motor_limits.py` 的 `compute_pd_torques`。SDK 使用 effort 模式，内部 stiffness/damping 清零，避免与外部 PD 叠加。力矩数组为 float32；相关 SDK 类型说明见 [NVIDIA 确认的文档修正](https://forums.developer.nvidia.com/t/documentation-fix/182527)。

物理步进后按 [IsaacGymEnvs 官方实现](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/base/vec_task.py) 调用阻塞 `fetch_results(sim, True)`，不把其返回值的真假作为跨绑定通用的成功条件。首个返回值记录在 JSON 的 `first_fetch_results_return` 中用于诊断；异常仍会传播，继续通过有限状态、越界和真实关节响应判断结果。修复了初版脚本在该接口返回假值时错误中断、计数停留在 0 的问题。

运行条件：

- 单机器人，车体固定在 z=1m，重力开启。
- 无地面，临时关闭自碰撞，先隔离验证关节和控制器；`create_actor` 使用共同过滤位 `1`，并读取每个碰撞 shape 的实际 filter 校验。源 URDF、网格、质量、惯量和关节定义不改动。
- 初始为用户确认的 J1=−25°、J2=+20°、轮关节=0。
- 每个腿关节分别施加 +5°/−5° 目标偏移；轮关节分别使用 +1/−1rad/s，再停止。
- 动作间恢复默认目标；全程 40 秒仿真时间、当前 heroleg26 默认 dt=0.001，即 40000 步。
- 无随机化、策略、接触或平衡测试。这里的正负响应只验证数值关节通道，实物方向仍依据用户已确认的坐标定义。

先只检查配置（不需要 SDK，也不写文件）：

```bash
python -B heroleg26_isaac_check/check_pd.py --repo "$PWD" --branch plane --check-config
```

服务器个人环境中，确认物理 GPU 2 空闲后执行：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -c 0
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/check_pd.py" \
  --repo "$HOME/projects/RM-wheelleg-rl" --branch plane --compute-device 0
```

后续跳跃分支可用 `--branch jump`；应先确认平地分支结果。

### 2026-11-09 服务器日志诊断

已读取 `pd-fixed-plane-20261109-194905-754154` 的完整 8000 行 CSV：两轮末段速度每一步翻转符号，约 +13.5/−13.5rad/s，力矩饱和占比分别约 99.875%/99.9125%；腿关节的目标和外部 PD 命令按正负测试正确切换，但实际平均角度几乎不变。仅凭此日志不能把腿部故障归咎于方向或关节限位。

初版入口误用 `collision_filter=-1`，这表示沿用资产的过滤设置，并不可靠地关闭自碰撞；旧报告的 `self_collisions_enabled=false` 只是脚本声明，没有读回验证，不能当成已关闭的证据。训练任务原本使用 `1`。已修正入口为 `1` 并读回 shape masks；物理步进后抽样检查接触数，无地面且自碰撞禁用的隔离检查应为零。SDK 过滤参数讨论见 [NVIDIA 论坛](https://forums.developer.nvidia.com/t/self-collision-avoiding-parent-links-or-links-higher-in-the-kinematic-chain/196626)。实际是否存在接触、该修复能否解释腿部受限，还需要服务器对照验证。

同时，单轮轴惯量约 0.000457893kg·m²，显式速度反馈 Kd=0.2、dt=0.005 的简化离散模型有 `Kd*dt/I≈2.184`，超过单惯量反馈稳定条件 0～2；预测饱和翻转速度约 13.45rad/s，与日志接近。这是隔离单轮的数值估算，不是完整多刚体证明，需控制周期对照。

上述对照已在服务器完成：

- `pd-fixed-plane-20261109-200055-655591`：filter=1、dt=0.005，腿部八项通过，轮部四项未通过。
- `pd-fixed-plane-20261109-200307-665946`：同样的过滤、增益、模型和限幅，dt=0.001，12 项全部通过，40000 步，状态 PASS。

验证依据是用户提供的终端输出；第二次运行的完整 JSON/CSV 尚未复制到本机，不声称已分析其全部瞬态或饱和情况。plane 固定车体检查通过，jump 分支尚未在服务器单独验证，落地接触、自由车体、实际电压范围及训练均未验证。

已将两个独立 heroleg26 配置的 sim.dt 设为 0.001；control.decimation 从继承的 2 改为 10，保持策略动作周期为 0.01s。原 wheel_legged 任务仍为 dt=0.005、decimation=2。该变更增加每个策略动作内的物理与 PD 更新次数，未改变机器人质量、惯量、限位或电机约束；计算量会增加。

`--dt` 继续仅用于单次检查的时间步对照，不回写配置。只增加物理 substeps 而不增加外部反馈更新频率，不等价于缩短外部 PD 控制周期。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/check_pd.py" \
  --repo "$HOME/projects/RM-wheelleg-rl" --branch plane --compute-device 0 --dt 0.001
```

入口重用已经校验的全分辨率 OBJ，不改写源文件。缓存位于 `~/.cache/heroleg26/`，每次运行的 CSV 和 JSON 位于独立的 `~/.local/state/heroleg26/isaacgym/pd-fixed-plane-时间戳/`；结束后销毁仿真。

## 输出判读

- `PASS`：按所选时间步完成 40 秒仿真（默认 40000 步），12 次正反响应均通过，末尾轮速回零且腿部回到默认站姿附近。仅是固定车体数值控制检查通过。
- `NEEDS_REVIEW`：完成物理步进，但方向、稳态误差或停止检查不达标。需分析 CSV、增益及动力学，不能直接判定 URDF 错误。
- `ERROR`：例如 SDK 失败、非有限状态或严重越界。尽可能保存报告，异常时也销毁 sim。

稳态窗口为每段末尾 0.25 秒；腿角目标误差容许 3°、轮速目标误差容许 0.2rad/s。阈值是本次诊断标准，不是实物精度指标。若出现中断或 native 崩溃，报告可能不完整，需保留终端输出。

CSV 的力矩列是**命令力矩**，不是力传感器读数。JSON 记录实际配置、源文件哈希、每段响应、最大速度、限幅占比和最终状态；保护源文件并非运行时只读权限机制。

固定测试成功后，再增加自由车体和地面接触检查；不能把本测试当成已实现两轮自平衡。当前约束仍是 24V 持续输出参考，不替代实际电压和热工况标定。

## 自由车体与平地接触检查

入口为 `check_contact.py`，复用上述配置读取、网格准备及外部 PD。当前本地没有 Isaac Gym，CPU 测试只能验证设置、数据解释和判定逻辑；实际 PhysX 结果需在服务器运行。

默认检查条件：

- 车体不固定；位置 `[0,0,0.265]`，用户确认站姿，所有初始速度为零。
- 物理/PD 时间步 0.001s，持续 5s（5000 步），无策略、训练、随机化或外部扰动。
- `--control hold` 用当前外部 PD 保持腿关节目标、轮速目标为零；不会根据车体倾角控制平衡。可用 `--control passive` 做零力矩对照。
- z=0 的水平地面，其静摩擦、动摩擦、恢复系数读取配置，目前都是 0.5；机器人 shape 的导入材料保持原值并记录，未进行实物摩擦标定。组合后的接触行为也不能仅凭地面系数确定。
- 自碰撞仍暂时关闭，公共过滤位为 1 并读取验证。地面接触保留。本阶段不验证自碰撞，后续如需启用应单独设计过滤规则与碰撞模型。
- 使用完整 OBJ 资源，但 PhysX 对网格的烹制方式未由本测试验证；不能据此宣称实际碰撞表面与每个 STL 三角面完全一致。

先检查配置（不使用 SDK/GPU、不写文件）：

```bash
python -B heroleg26_isaac_check/check_contact.py --repo "$PWD" --branch plane --check-config
```

服务器个人 Isaac Gym 环境中，确认 GPU 2 空闲后执行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/check_contact.py" \
  --repo "$HOME/projects/RM-wheelleg-rl" --branch plane --compute-device 0
```

`--seconds` 可改变持续时间；`--lift` 可增加 0～0.2m 的诊断释放高度，默认不增加。入口要求 1ms 或更小时间步，避免误用已发现会振荡的旧配置；上述选项都不回写源文件。

### 接触结果与日志

新日志写入 `~/.local/state/heroleg26/isaacgym/contact-plane-时间戳/`，缓存仍在个人目录。`report.json` 记录初次接触、每轮接触、接触力、关节限幅、倾倒和源文件哈希；`trace.csv` 逐步记录车体姿态/速度、各关节角度/速度、接触体、各体净接触力 z 分量及命令力矩。

CSV 时间是完成物理步后的时刻，状态和接触力取自该时刻；`commanded_tau_pre_step_Nm_*` 是这一步开始时施加的外部命令，不是电机传感器测量值。关节反馈速度采用 SDK 原始 STATE_ALL 值，与已通过的固定车体检查一致；训练任务中的位置差分估速仍需在训练环境检查。

接触 API 优先按命名字段解析，不猜测无名称 tuple 的内存顺序；若接触格式没有命名字段，会保留类型与首条数据预览，并通过每体净接触力辅助判定。在单机器人、已验证关闭自碰撞且只存在地面的条件下，每体向上接触力可归于地面。

- `PASS`：完成请求时长、状态有限、观察到地面接触及两个轮子的地面接触，两个轮子各自至少出现过超过 1N 的向上接触力，且没有触发下述诊断阈值。它只表示基础接触和数值检查通过，不表示能平衡、能行驶或接触网格完全准确。
- `Fall observed: True`：车体倾角达到过 60°，属于无平衡策略情况下的观察结果，单独不判失败。
- `NEEDS_REVIEW`：例如未观察到轮地接触、接触承载、出现自碰撞、严重关节越界或可疑穿地，需要分析日志。
- `ERROR`：初始化或 API 异常、非有限状态、严重穿地/发散等；异常时尽可能保存报告并销毁 sim。

诊断阈值为：base_link 原点低于地面 5cm、车体速度超过 20m/s、关节速度超过工作上限 4 倍、膝角超出源限位 0.035rad、SDK initial_overlap 超过 5cm，均需复核。严重穿地至 −0.5m 或车速超过 100m/s 会提前中止。这些是排查阈值，不是实物允许值，也不是精确的碰撞穿透测量。

通过后仍需检查承载状态的 PD/动作范围、实际训练环境、域随机化和奖励，再开始小规模平地平衡训练。此入口不自动提交或推送 Git，也不修改系统文件。

## 机身触地后的平衡训练失败规则

2026-11-10 的接触日志虽通过基础接触检查，但约 0.152s 开始机身承重；末 1s 的机身接触力均值约 29.18N、两侧 J2 接近 50° 上限、车体倾角约 15.13°。0.2169m 是此支撑状态下的车体原点高度，不能作为正常双轮站立的标定高度。

`plane` 和 `jump` 下的 `Heroleg26Cfg` 现明确设置：

- `asset.penalize_contacts_on = ["base_link"]`，复用现有碰撞惩罚；现有函数以接触合力模长 >0.1N 判断，未调整奖励权重。
- `asset.terminate_after_contacts_on = ["base_link"]`，复用现有接触失败规则，接触合力模长 >10N。
- `env.fail_to_terminal_time_s = 0.05`。现有失败计数必须连续，失败解除即清零；100Hz 动作频率下一般持续约 50～60ms 触发重置。该计数也用于既有的接近翻倒判据，缩短了其重置延迟。

这套规则针对初期平衡/行驶训练，后续若训练翻倒自救，必须设计独立任务。瞬时冲击不会单独触发持续接触失败；轮子接地不在惩罚/终止列表。原始 `wheel_legged` 任务配置及通用终止函数保持原样，URDF、质量、惯量、机械限位、默认站姿和默认 PD 增益保持原样。

### 实际训练环境中的规则验证

新增 `check_training_env.py`，默认选择物理 GPU 1、创建 4 个环境、执行 500 个零动作步。它不创建算法 runner，不加载策略，不执行 PPO。

```bash
python -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/check_training_env.py" \
  --repo "$HOME/projects/RM-wheelleg-rl" --check-config
```

在个人 Conda 环境中，设置好 `LD_LIBRARY_PATH` 后运行：

```bash
python -u -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/check_training_env.py" \
  --repo "$HOME/projects/RM-wheelleg-rl" --gpu 1
```

入口内部选择 GPU；不必同时设置另一个 `CUDA_VISIBLE_DEVICES`。进程内使用 `cuda:0`。它临时关闭域随机化、观测噪声和速度指令，将单回合缩短到 2s；保留真实任务固有的随机初始根速度，报告对此有明确说明。这不是完全静止初始化的独立接触检查。

判据检查使用真实任务的 `check_termination`，通过只读观察包装器在 `reset_idx` 覆盖状态前克隆终止状态，避免把重置后的高度误当成触地时的高度。

- `Environment contact-rule check: PASS`：有限状态、观测/历史/特权维度正确，并实际观察到机身触地碰撞惩罚和非超时、非接近倒置的机身触地重置。
- `NEEDS_REVIEW`：没有触发预期机身惩罚/重置；不一定代表仿真发散，应检查轨迹和增益变化。
- `Height reference: NOT_CALIBRATED`：本入口不自动确定训练高度。另记倾角 <5°、双轮分别 >1N 承重、机身接触合力 <1N、腿目标误差 <5°、根线速度 <0.05m/s 的候选样本及最长持续时长，供后续复核。

`--leg-kp-scale` / `--leg-kd-scale`（0.25～4，默认 1）仅供后续短时控制对照；只缩放四个腿关节的 Kp/Kd，不调整轮子增益、力矩/转速上限或源文件。默认先按原增益验证规则，不要以加大增益代替平衡策略。

报告/逐步终止前轨迹写在个人 `~/.local/state/heroleg26/isaacgym/env-contacts-plane-时间戳/`，编译和绘图缓存均在个人 `~/.cache/heroleg26/`。本机 CPU 测试不能代替服务器 PhysX 运行。

独立 `check_contact.py` 现在还会打印 `Wheel-only support`、非轮部件承重列表和首次时刻；其原来的基础 `PASS` 定义仍保持原样，不能与正常站立混用。

## 100 轮试跑后的独立评估与速度重置修复

完整的 100 轮日志显示：5002 次训练重置中，5000 次记录为机身触地相关、2 次为超时；归一化动作裁剪比例约 0.0756%。这不是确定性策略的评估成功率，动作裁剪比例也不等同于力矩或膝目标的饱和比例。

代码复现发现：`compute_dof_vel()` 启用位置差分时会令 `self.dof_vel` 指向另一个计算张量。旧 `_reset_dofs()` 清零这个张量后，传入 `set_dof_state_tensor_indexed()` 的原生 `dof_state[...,1]` 仍可能带着旧关节速度。`plane` / `jump` 的重置现同时清零原生状态和反馈速度。修复只作用于回合初始化，不改变运动中的速度估计、质量、惯量、限位、网格、站姿或增益；它也作用于共享此重置函数的原始任务。已有 100 轮模型无需删除，但不能把旧训练日志当作修复后训练的结果，也不能把所有失败都归因于这一问题。

`evaluate_balance.py` 读取指定训练目录中的 `effective_config.json` 与 `model_100.pt`，保留训练时的资产、站姿、增益、动作裁剪和地面设置，调用 `act_inference(obs, history)` 的动作均值，不采样策略噪声、不运行 PPO。

- 比较 `zero_actions`（腿关节默认站姿、轮目标零速的原 PD）与 `policy_mean`。
- 每种方式默认运行一回合零初速度，以及 8 回合固定种子的初始速度；根线速度与角速度各分量均在训练固有的 −0.5～0.5 范围内。两种方式使用完全相同的初始速度列表，但不保证 GPU 结果逐位相同。
- 使用单环境顺序测试，默认每回合最长 5s；保留真实失败重置规则。每次开始通过 SDK 设置根初速度，同时验证原生关节速度已清零。标准 reset 含一个共同的零动作预热步，随后用当前观测填充历史，避免带入上一试验的历史。
- 在自动重置前保存状态，记录终止原因、时长、最大倾角、水平漂移、机身承重、其他非轮部件承重、双轮承重占比、膝关节接近限位、命令力矩饱和、动作裁剪等。
- `clean_balance` 是初步诊断判据：达到完整时长，无非轮部件向上接触力 >1N，至少 95% 采样点双轮各承受 >1N，最大倾角 ≤10°、最大水平位移 ≤0.25m。不是赛事、实物或正式安全标准。
- 输出 `Evaluation: COMPLETED` 仅表示评估跑完；是否符合上述平衡判据要看每组 `clean_balance_trials`，不会把流程完成打印成平衡 PASS。
- 入口核对训练源文件和物理资源哈希，只允许通用环境源代码相对训练快照多出上述原生关节速度清零语句。其他不一致会停止，避免用不同模型或控制代码评估旧策略。

服务器激活个人环境并设置 `LD_LIBRARY_PATH` 后，先配置检查再运行（替换为实际训练目录）：

```bash
python -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/evaluate_balance.py" \
  --run "$HOME/.local/state/heroleg26/isaacgym/ppo-smoke-plane-20261110-143041-434778" --check-config
python -u -B "$HOME/projects/RM-wheelleg-rl/heroleg26_isaac_check/evaluate_balance.py" \
  --run "$HOME/.local/state/heroleg26/isaacgym/ppo-smoke-plane-20261110-143041-434778" --gpu 1
```

结果保存到个人 `~/.local/state/heroleg26/isaacgym/eval-balance-时间戳/`，含报告、两个控制方式的 CSV 和逐回合摘要。初始状态重置修复与历史清洁初始化会使本次评估和旧训练轨迹不完全相同；这是对当前模型的独立诊断，不是旧随机训练的逐步重放。本机没有 Isaac Gym，只验证了 CPU 逻辑、实际模型加载和确定性前向计算；服务器实测仍需运行。
