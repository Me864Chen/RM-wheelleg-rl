#!/usr/bin/env python3
"""Prove visual additions preserve collision shapes, dynamics and policy rollout."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np
from verify_rm2026_scene import verify,policy_rollout
ROOT=Path(__file__).resolve().parents[1]

def verify_visual(source,scene,rollout=False):
    a=mujoco.MjModel.from_xml_path(str(source));b=mujoco.MjModel.from_xml_path(str(scene))
    assert a.stat.extent<100 and b.stat.extent<100,'Invalid rendering bounds'
    assert a.nq==b.nq and a.nv==b.nv and a.nu==b.nu
    for attr in ['body_mass','body_inertia','body_ipos','body_iquat','dof_damping','dof_armature','jnt_range','actuator_gear','actuator_ctrlrange']:
        np.testing.assert_array_equal(getattr(a,attr),getattr(b,attr),err_msg=attr)
    assert a.opt.timestep==b.opt.timestep
    checked=0
    fields=['geom_type','geom_contype','geom_conaffinity','geom_condim','geom_priority','geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap']
    for i in range(a.ngeom):
        name=mujoco.mj_id2name(a,mujoco.mjtObj.mjOBJ_GEOM,i)
        if name:
            j=mujoco.mj_name2id(b,mujoco.mjtObj.mjOBJ_GEOM,name)
        else:
            bid=a.geom_bodyid[i];body=mujoco.mj_id2name(a,mujoco.mjtObj.mjOBJ_BODY,int(bid))
            other=mujoco.mj_name2id(b,mujoco.mjtObj.mjOBJ_BODY,body)
            assert other>=0 and a.body_geomnum[bid]==b.body_geomnum[other]
            j=int(b.body_geomadr[other]+i-a.body_geomadr[bid])
        assert j>=0,name
        for field in fields:np.testing.assert_allclose(getattr(a,field)[i],getattr(b,field)[j],atol=1e-9,rtol=1e-9,err_msg=f'{name}:{field}')
        if a.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            ai=a.geom_dataid[i];bi=b.geom_dataid[j];av=a.mesh_vertadr[ai];bv=b.mesh_vertadr[bi];n=a.mesh_vertnum[ai]
            assert n==b.mesh_vertnum[bi]
            np.testing.assert_array_equal(a.mesh_vert[av:av+n],b.mesh_vert[bv:bv+n])
        checked+=1
    added=0
    for i in range(b.ngeom):
        name=mujoco.mj_id2name(b,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if name.startswith('vis_'):
            assert b.geom_contype[i]==0 and b.geom_conaffinity[i]==0,name
            added+=1
    result={'existing_geometries_unchanged':checked,'noncolliding_visual_geometries':added,'robot_dynamics':'identical','physical_checks':verify(scene,ROOT/'scenes/rm2026_parameters.json')}
    if rollout:result['policy_rollout']=policy_rollout(scene)
    result['status']='passed';return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rollout',action='store_true');p.add_argument('--report',type=Path)
    a=p.parse_args();r=verify_visual(ROOT/'scenes/rm2026_arena.xml',ROOT/'scenes/rm2026_visual.xml',a.rollout)
    text=json.dumps(r,indent=2,ensure_ascii=False);print(text)
    if a.report:a.report.write_text(text+'\n')
if __name__=='__main__':main()
