# Isaac Gym 单机器人检查

全部操作使用普通用户环境。已有 `check_isaacgym.py` 只创建 actor、不步进物理；新 `check_pd.py` 执行真正的固定车体 PD 检查，不加载策略、不训练、不创建 Viewer。

## 固定车体 PD 检查

从当前仓库的 base → wheel_legged → heroleg26 配置读取站姿、PD、限幅和时间步。入口只解析所需的字面量赋值，不导入完整 RL 包；遇到不支持的配置表达式会报错，不猜测其含义。

训练环境与检查入口共用所选分支 `utils/motor_limits.py` 的 `compute_pd_torques`。SDK 使用 effort 模式，内部 stiffness/damping 清零，避免与外部 PD 叠加。力矩数组为 float32；相关 SDK 类型说明见 [NVIDIA 确认的文档修正](https://forums.developer.nvidia.com/t/documentation-fix/182527)。

物理步进后按 [IsaacGymEnvs 官方实现](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/base/vec_task.py) 调用阻塞 `fetch_results(sim, True)`，不把其返回值的真假作为跨绑定通用的成功条件。首个返回值记录在 JSON 的 `first_fetch_results_return` 中用于诊断；异常仍会传播，继续通过有限状态、越界和真实关节响应判断结果。修复了初版脚本在该接口返回假值时错误中断、计数停留在 0 的问题。

运行条件：

- 单机器人，车体固定在 z=1m，重力开启。
- 无地面，临时关闭自碰撞，先隔离验证关节和控制器；`create_actor` 使用共同过滤位 `1`，并读取每个碰撞 shape 的实际 filter 校验。源 URDF、网格、质量、惯量和关节定义不改动。
- 初始为用户确认的 J1=−10°、J2=+40°、轮关节=0。
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
