# heroleg26 当前仿真模型：4.SLDASM

本目录现使用用户重新导出的 `4.SLDASM` 模型。原始副本位于 `../heroleg26/urdf/4.SLDASM.urdf`，网格位于 `../heroleg26/meshes/`。项目中旧的原始 URDF、CSV、网格和导出元数据已替换；下载目录中的源文件不修改。

## 当前文件

- `urdf/heroleg26.urdf`：新导出的完整模型，将 14 处 `package://4.SLDASM/meshes/` 引用改为 `../meshes/`，其余 XML 数据保留。
- `urdf/heroleg26_visual.urdf`：MuJoCo 检查版；主车体外观使用简化 OBJ，碰撞使用新导出的全分辨率 OBJ，其余连杆沿用新 STL。
- `meshes/*.STL`：新导出的 7 个原始网格副本。
- `meshes/visual/base_link_visual.obj`：主车体仅外观简化，2,091,770 面降至 104,588 面。
- `meshes/collision/base_link_full.obj`：新主车体的全部 2,091,770 个三角面，无碰撞简化。
- `../heroleg26_isaac_check/generated/heroleg26_obj.urdf`：由现有准备工具生成的 Isaac Gym OBJ 版；7 个网格都保留全部三角面。这是可重新生成的本机缓存，不纳入 Git。

两份 MuJoCo OBJ 已在 `.gitignore` 中单独允许保存，避免克隆后检查入口缺少资源。物理属性来自新导出，不能将旧版的质量、零位、转轴或检查报告沿用到此版本。

## 新导出的已知定义

- 总质量：约 **13.38048145 kg**，直接汇总新 URDF 的惯性质量。
- J1、J3：`continuous`，源文件未提供 effort/velocity。
- 左右 J2：`revolute`，均为 `[-0.41538836197465, 0.872664625997165] rad`；源文件给出 effort=30、velocity=1000。
- 上述电机数值仅作为源文件保留；独立训练任务已接入 24V 参考持续输出配置，运行时覆盖 SDK 电机属性，详见 [ACTUATORS.md](ACTUATORS.md)。实际电压范围内的输出能力仍待标定。
- 车体坐标已随重新导出变化：前方为 +X，左侧为 +Y，上方为 +Z。依据为车体质量、惯量与质心在原坐标到新坐标的 -90° 水平旋转关系，并结合两侧髋关节位置核对。
- 用户已确认训练默认站姿：两侧 J1=−10°、J2=+40°，轮关节为 0°；见 [CALIBRATION.md](CALIBRATION.md)。独立检查工具仍从 URDF 零位启动。

## 检查入口

```bash
source /home/mcsw/RL_infantry/.venv-mujoco/bin/activate
python -E /home/mcsw/RL_infantry/RM-wheelleg-rl/heroleg26_sim/tools/inspect_model.py
```

静态入口不积分动力学、不加载 ONNX，也不回写源文件。按键见 [INSPECTION.md](INSPECTION.md)。训练坐标及运动学适配见 [CALIBRATION.md](CALIBRATION.md)，通道对应见 [JOINT_MAPPING.md](JOINT_MAPPING.md)。

## 验证记录

- `resource_manifest.json`：当前源 URDF、7 个网格的路径和哈希。
- `mesh_preparation_report.json`：新主车体外观简化记录。
- `collision_geometry_check.json`：所有 STL 三角面到 OBJ 的逐项验证。
- `physics_preservation_check.json`：新模型的全分辨率基线与外观简化版，MuJoCo 物理数组和 7 个碰撞网格一致。
- `model_replacement_report.json`：导入来源、新关节定义、坐标转换依据和检查姿态。

这些记录只验证本次资源替换、静态导入与运动学，不表示已完成电机标定、Isaac Gym 动力学或训练验证。当前训练中涉及的接触设置、控制增益、动作范围等仍需审查。
