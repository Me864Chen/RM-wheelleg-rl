#!/usr/bin/env python3
"""Run visual RM2026 scene; existing run_rm2026.py keeps the motion prototype."""
from pathlib import Path
import onnx_mj_chuanlian as controller

def main():
    scene=Path(__file__).resolve().parents[1]/'scenes/rm2026_visual.xml'
    if not scene.exists():raise SystemExit('Run python -E python_tools/build_rm2026_visual.py first.')
    for preset in controller.RUNTIME_PRESETS.values():preset['xml_path']=str(scene)
    print('[SCENE] RM2026 visual approximation; physics inherited from motion scene. See scenes/README_rm2026_visual.md',flush=True)
    controller.main()
if __name__=='__main__':main()
