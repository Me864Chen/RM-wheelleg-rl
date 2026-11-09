# 训练关节映射

`plane` 和 `jump` 使用相同的逻辑通道：

| 策略通道（从 0 开始） | 开源模型关节 | heroleg26 关节 |
| --- | --- | --- |
| 0 | lf0_Joint | RJ1 |
| 1 | lf1_Joint | RJ2 |
| 2 | l_wheel_Joint | RJ3 |
| 3 | rf0_Joint | LJ1 |
| 4 | rf1_Joint | LJ2 |
| 5 | r_wheel_Joint | LJ3 |

此处“左/右”沿用开源控制器的逻辑角色，按用户指定的对应关系绑定；不能根据 LJ/RJ 字母重新交换。
开源 URDF 中轮关节的准确拼写是 `l_wheel_Joint`、`r_wheel_Joint`。

## 代码位置

两个训练目录下各有独立的 `wheel_legged_gym` 包，修改已同步：

- `envs/wheel_legged/wheel_legged_config.py`：`asset.policy_joint_names` 定义策略顺序，`asset.joint_name_mapping` 定义旧名称别名；默认角度及 PD 增益改用 RJ/LJ 名称，数值按对应关系继承。
- `utils/joint_mapping.py`：按加载后的名称查找索引；缺失、重复或不匹配的关节会报错。
- `envs/base/legged_robot.py`：动作转为 PhysX 原生顺序；观测转回策略顺序；腿部运动学和关节奖励使用同一映射。PhysX 的状态、力矩、限位、复位缓冲区仍保持原生顺序。

例如 SDK 返回 `[LJ1,LJ2,LJ3,RJ1,RJ2,RJ3]` 时，策略到 SDK 的索引为 `[3,4,5,0,1,2]`。
RJ1/RJ2/LJ1/LJ2 继承位置 PD 控制通道；RJ3/LJ3 继承轮速控制通道。启动环境时会打印每个策略通道对应的实际关节及索引。

## 站姿和运动学适配进展

当前已换为新导出的 4.SLDASM，车体前方为 +X、顶部为 +Z；训练默认站姿已确认为两侧 J1=−10°、J2=+40°，轮关节为 0°。两分支使用独立 `heroleg26` 任务。详见 [CALIBRATION.md](CALIBRATION.md)。RJ 为实际右侧、LJ 为实际左侧，原控制通道对应关系继续保留。

原 `wheel_legged` 任务保留原模型文件、原默认角度、原运动学；新 `heroleg26` 任务使用自己的资源路径、坐标转换和从 URDF 推导的腿部几何，不将两种模型的参数混用。两者均按实际 SDK DOF 名称绑定索引。

模型物理参数按新导出保留。J2 的源限位已导入，电机输出端限制尚未完成厂家数据标定或动力学验证。旧 MuJoCo 演示、旧 ONNX 和历史日志保持原样。

## 本地检查

无需 Isaac Gym，使用 CPU PyTorch：

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

包含关节乱序、单通道控制和新站姿/运动学检查。若安装了 MuJoCo，还会自动与实际导入模型做随机姿态对比。它不能代替服务器 Isaac Gym 动力学验证。
