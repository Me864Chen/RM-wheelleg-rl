# RM2026 场地视觉版（第一版）

依据《RoboMaster 2026 机甲大师超级对抗赛比赛规则手册 V2.2.0（20260807）》第4章的图4-1、4-2、4-5、4-11、4-20、4-21、4-24制作。

在现有运动地形上，增加深灰 PVC 地胶、浅灰高地材质、红蓝分区线、白色功能区轮廓、停机区八边形标线、道路边线、方向箭头、警示条纹、基地装甲外壳、资源站外壳、中央机关静态示意和围挡外观。地面纹理由脚本确定性生成，不需要下载资源或新增 Python 包。

这是一版接近手册风格的视觉近似场景，尚非完整 CAD 复原。现有运动原型中的平台形状、功能区相对位置仍有估算；标线的位置和配色也来自插图估计，不可作为毫米级测量结果。前哨站、飞镖发射站、雷达、标志字样及其他比赛设施尚未完整建模。中央机关目前仅作静态机械外形示意，不复现比赛动作和规则。视觉标线不是完整的裁判判定区域数据。

## 启动

在任意终端运行：

```bash
source /home/mcsw/RL_infantry/.venv-mujoco/bin/activate
cd /home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco
python -E /home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/run_rm2026_visual.py --device cpu
```

按键沿用现有控制：长按1/2累加前后目标速度，长按3/4累加角速度；松开相应轴的按键后该轴目标速度归零；7停车，0切换第一/第三人称。默认 GLFW 查看器无原生侧栏。如需原生侧栏，在命令末尾加 `--keyboard-backend viewer`，但该模式为按键持续运动、7停车，不支持可靠的松开检测。

## 回到此前运动场景

退出当前窗口后运行：

```bash
python -E /home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/run_rm2026.py --device cpu
```

运动原型 `/home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/scenes/rm2026_arena.xml` 和原始机器人场景均未被这次修改覆盖。切换启动入口即可回退，无须撤销 Git 工作区内的其他修改。

## 碰撞范围

视觉版继承全部原有碰撞几何、坡度、高差、沟壑、通道净空、摩擦系数、机器人质量/惯量、策略和相机。新增标线与装饰均设置 `contype=0, conaffinity=0`，不额外阻碍机器人。

基地和补给设施仍使用此前运动版的简化碰撞包络，视觉外壳与真实外形不完全一致。中央新增的悬空环、辐条与装饰外壳不参与碰撞；原有两根支撑的碰撞保持不变。因此不能用这一版评估机器人与这些装饰细部的碰撞。若后续需要撞击机关、前哨站等，应另行完成与外形一致的碰撞模型和验证。

2.4m 围挡的原碰撞边界保留。视觉上改用低位黑色挡板、上部透明围挡及细框；透明部分也保留原有碰撞作用。

## 修改、再生成与验证

主要代码：

- `/home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/build_rm2026_visual.py`：颜色、材质、标线和设施外形。
- `/home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/run_rm2026_visual.py`：独立启动入口。
- `/home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/verify_rm2026_visual.py`：检查碰撞模型、机器人动力学和策略回放。
- `/home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/scenes/rm2026_visual.xml`：生成的视觉场景。
- `/home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/scenes/rm2026_visual_manifest.json`：来源与近似项。

更新运动场景后，需要重新生成视觉版：

```bash
python -E /home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/build_rm2026_visual.py
python -E /home/mcsw/RL_infantry/RM-wheelleg-rl/mujoco/python_tools/verify_rm2026_visual.py --rollout
```

已验证：新旧场景全部1059个原有几何的碰撞参数及网格顶点一致；1571个新增视觉几何不参与碰撞；机器人动力学参数一致；16段坡面、650mm沟壑、700mm通道宽度、260mm净空、落体接触和通道通过/阻挡测试通过。20秒原策略平地回放通过，行驶约1.918m，无数值警告。这不代表已经验证跨越所有障碍；也尚未测量本机桌面交互帧率。
