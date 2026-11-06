#!/usr/bin/env python3
"""Run the separate RM2026 motion scene with existing policy and keyboard controls.

Original scene: python -E python_tools/onnx_mj_chuanlian.py --device cpu
RM2026 scene:   python -E python_tools/run_rm2026.py --device cpu
"""
from pathlib import Path
import onnx_mj_chuanlian as controller

def main():
    scene=Path(__file__).resolve().parents[1]/'scenes'/'rm2026_arena.xml'
    if not scene.exists():
        raise SystemExit('Scene missing. Run: python -E python_tools/build_rm2026_scene.py')
    for preset in controller.RUNTIME_PRESETS.values():
        preset['xml_path']=str(scene)
    print('[SCENE] RM2026 motion prototype. Dimension assumptions: scenes/README_rm2026.md',flush=True)
    controller.main()

if __name__=='__main__':main()
