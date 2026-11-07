# heroleg26 仿真资源副本

此目录从 `../heroleg26` 复制，用于后续仿真适配。原始导出目录未修改。

```text
heroleg26_sim/
├── urdf/heroleg26.urdf
├── meshes/                 # 7 个独立复制的 STL 文件
├── resource_manifest.json # 路径检查结果、文件大小与 SHA-256
└── README.md
```

## 本次改动

- 将原始 `urdf/2.SLDASM.urdf` 复制为 `urdf/heroleg26.urdf`。
- 将全部 14 处网格引用从 `package://2.SLDASM/meshes/...` 改为 `../meshes/...`，按 URDF 文件所在目录解析。
- 网格使用真实副本，不是软链接；移动整个目录时无需依赖原始目录。
- 保留机器人内部名称 `2.SLDASM`，所有连杆、关节、质量、惯量、轴向、零位、限位和颜色保持原值。
- 没有复制旧 ROS 包的 package.xml、CMakeLists.txt 和 launch 文件，本目录是独立模型资源，不是已配置的 ROS 包。

## 已检查

14 处网格引用均指向本目录内存在的文件；7 个 STL 副本与源文件哈希一致；URDF 除网格路径外结构完全一致；原始目录全部文件哈希未改变。

## 尚未完成

本次只整理资源，不代表已经能直接在 MuJoCo 或 Isaac Gym 中运行。主车体 STL 仍保留原始约 209 万个三角面，已知超出本机 MuJoCo 3.2.7 的 STL 读取面数上限。原始版本仍未减面；本次另增加仅简化外观的版本，详见下节。未制作近似碰撞几何、补充电机参数或生成正式 MJCF。

MuJoCo 的 URDF 导入还需要明确配置资源路径解析，例如转换时设置合适的 meshdir/strippath；不能假定所有工具都按相同规则解析相对路径。后续适配会在转换入口处理该问题。

下一步可处理过大的车体网格并准备碰撞模型，再检查关节方向与零位。后续仿真修改优先在本目录进行。


## 仅简化外观的可选版本（物理参数保留）

新增 `urdf/heroleg26_visual.urdf`，原有 `urdf/heroleg26.urdf` 保留。新版本只有主车体的两处资源引用不同：

- visual → `../meshes/visual/base_link_visual.obj`：显示网格由 2,091,770 面减至 104,588 面。细小外观细节可能有变化。
- collision → `../meshes/collision/base_link_full.obj`：全分辨率碰撞网格，从 STL 转为 OBJ；保留全部 2,091,770 个三角面和原始坐标，绕过 MuJoCo 3.2.7 的 STL 读取面数限制。

其余六个连杆的外观与碰撞资源保持原值。所有质量、质心、惯量、关节类型/坐标/转轴/零位、限位、effort、velocity、颜色及比例保持原值，没有根据简化外观重算任何物理量。尤其原有 effort/velocity=0 的未配置状态也保留，后续仍需另行适配训练。

### 已完成的保留性检查

1. 新 URDF 除主车体 visual/collision 的文件引用外，与基线 URDF 数据完全一致。
2. 7 个原始 STL、基线 URDF、源导出 URDF 的 SHA-256 与之前保存值一致。
3. 全分辨率碰撞 OBJ 逐三角面与 STL 对比，所有顶点在原始 float32 精度下完全一致。
4. MuJoCo 3.2.7 中，对全分辨率基线与简化外观版本进行编译对比：body_mass/body_inertia/质心/关节参数等数组完全一致，7 个碰撞几何及其编译后的网格顶点完全一致。

编译对比中的基线使用同一份全分辨率 OBJ 替代超出读取上限的 base STL；两边均在内存中添加相同的临时浮动根节点并设置相同资源解析参数，仅用于检查，未写入正式 URDF。此检查不是 Isaac Gym 验证或行走/跳跃策略测试。真实可用接触面仍由仿真器的网格碰撞处理方式决定，本次未调整这些设置。

检查报告：`mesh_preparation_report.json`、`collision_geometry_check.json`、`physics_preservation_check.json`。

此方案减轻显示网格，碰撞仍使用精细全分辨率几何，因此不承诺碰撞计算显著加速。为遵守物理量保留要求，没有替换成大盒子、圆柱或凸分解近似模型。


## 独立模型检查入口

新增 `tools/inspect_model.py`，支持选择六个关节、小角度增减、回零、关节轴及碰撞外形显示。使用原生 MuJoCo 侧栏，静态检查不积分动力学。启动和按键见 [INSPECTION.md](INSPECTION.md)。
