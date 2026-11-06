#!/usr/bin/env python3
"""Physical geometry and collision tests for the generated RM2026 scene.

Uses the MuJoCo engine, not an image comparison. No desktop or renderer required.
"""
from pathlib import Path
import argparse,json,math,xml.etree.ElementTree as ET
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]


def verify(scene,parameters):
    p=json.loads(parameters.read_text());m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    # Isolate environment for rays without changing collision physics.
    for i in range(m.ngeom):
        m.geom_group[i]=0 if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('rm_') else 5
    groups=np.array([1,0,0,0,0,0],dtype=np.uint8)
    def ray(point,direction):
        hit=np.zeros(1,dtype=np.int32)
        distance=mujoco.mj_ray(m,d,np.array(point,dtype=float),np.array(direction,dtype=float),groups,1,-1,hit)
        return distance,int(hit[0])
    ground=lambda y:-abs(y)*math.tan(math.radians(p['field']['crossfall_deg']))
    def surface(x,y):
        distance,_=ray([x,y,5],[0,0,-1]);assert distance>=0,(x,y);return 5-distance
    checks={}
    # Arena floor retains crossfall, not an infinite flat plane beneath the features.
    for y in [-6,-2,2,6]:
        np.testing.assert_allclose(surface(-12.0,y),ground(y),atol=2e-5)
    checks['floor_crossfall_deg']=p['field']['crossfall_deg']
    # Clear down to the real base floor inside the 650mm gap; no convex hull bridge.
    f=p['fly'];x,y=f['red_lip_center'];gap=f['gap']
    np.testing.assert_allclose(surface(x-gap/2,y),ground(y),atol=2e-5)
    np.testing.assert_allclose(surface(x-gap-.03,y),ground(y)+f['landing_height'],atol=2e-5)
    run=(f['lip_height']-f['tail_height'])/math.tan(math.radians(f['slope_deg']))
    for t in [.1,.5,.9]:
        np.testing.assert_allclose(surface(x+run*t,y)-ground(y),f['lip_height']+(f['tail_height']-f['lip_height'])*t,atol=2e-5)
    checks.update(fly_gap_m=gap,fly_angle_deg=f['slope_deg'],fly_lip_m=f['lip_height'],fly_tail_m=f['tail_height'])
    # Tunnel floor/ceiling and lateral walls are independently intersectable.
    t=p['tunnel'];cx,cy=t['red_center'];floor=ground(cy)+p['road']['deck_height']
    distance,_=ray([cx,cy,floor+.001],[0,0,1]);np.testing.assert_allclose(distance+.001,t['clear_height'],atol=2e-5)
    # Crossfall tilts the reference plane; compare horizontal distance at its centre height.
    distance,_=ray([cx,cy,floor+.10],[0,1,0]);np.testing.assert_allclose(distance,t['inner_width']/2,atol=2e-5)
    distance,_=ray([cx,cy,floor+.10],[0,-1,0]);np.testing.assert_allclose(distance,t['inner_width']/2,atol=2e-5)
    checks.update(tunnel_width_m=t['inner_width'],tunnel_clearance_m=t['clear_height'])
    # Verify all nominal ramp profiles in the actual compiled collision geometry.
    manifest=json.loads(scene.with_name('rm2026_manifest.json').read_text());count=0
    for ramp in manifest['features']:
        if ramp['kind']!='ramp' or 'rough_' in ramp['name'] or 'guard' in ramp['name']:continue
        ox,oy=ramp['origin'];heading=ramp['heading'];L=ramp['length']
        # Approach ramps can meet an adjoining deck at the endpoints; test interior.
        for v in [.35,.65]:
            qx=ox+L*v*math.cos(heading);qy=oy+L*v*math.sin(heading)
            expected=ramp['start']+(ramp['end']-ramp['start'])*v
            np.testing.assert_allclose(surface(qx,qy)-ground(qy),expected,atol=3e-5,err_msg=ramp['name'])
        count+=1
    checks['ramp_profiles_checked']=count
    # Sphere drop: contact must be generated, finite and supported, not merely visible.
    tree=ET.parse(scene);world=tree.getroot().find('worldbody')
    probe=ET.SubElement(world,'body',name='validation_probe',pos=f'-5 -2 {ground(-2)+.3}')
    ET.SubElement(probe,'freejoint',name='validation_probe_free')
    ET.SubElement(probe,'geom',name='validation_probe_geom',type='sphere',size='.04',mass='.1',contype='1',conaffinity='2',friction='1 .005 .0001')
    # Make copied asset references independent of the current working directory.
    for e in tree.getroot().find('asset'):
        if e.get('file'):e.set('file',str((scene.parent/e.get('file')).resolve()))
    pm=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));pd=mujoco.MjData(pm)
    probe_id=mujoco.mj_name2id(pm,mujoco.mjtObj.mjOBJ_BODY,'validation_probe');pg=mujoco.mj_name2id(pm,mujoco.mjtObj.mjOBJ_GEOM,'validation_probe_geom')
    hits=0
    for _ in range(300):
        mujoco.mj_step(pm,pd)
        hits+=sum(pg in (c.geom1,c.geom2) for c in pd.contact)
        assert np.isfinite(pd.qpos).all()
    assert hits>0,'No real probe-ground contact'
    assert pd.xpos[probe_id,2]>ground(pd.xpos[probe_id,1])+.03,'Probe fell through floor'
    checks['probe_contact_samples']=hits
    # The same opening must pass a small body and physically block a taller one.
    # A single slide joint removes tipping, so the test measures passage geometry.
    passage_results={}
    for label,half_height in [('fits',.04),('blocked',.18)]:
        trial=ET.parse(scene);root=trial.getroot();root.find('option').set('gravity','0 0 0')
        for e in root.find('asset'):
            if e.get('file'):e.set('file',str((scene.parent/e.get('file')).resolve()))
        world=root.find('worldbody');body=ET.SubElement(world,'body',name='clearance_probe',pos=f'{cx-.6} {cy} {floor+max(half_height+.01,.12)}')
        ET.SubElement(body,'joint',name='clearance_slide',type='slide',axis='1 0 0',damping='0')
        ET.SubElement(body,'geom',name='clearance_probe_geom',type='box',size=f'.06 .1 {half_height}',mass='.1',contype='1',conaffinity='2')
        tm=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));td=mujoco.MjData(tm)
        jid=mujoco.mj_name2id(tm,mujoco.mjtObj.mjOBJ_JOINT,'clearance_slide');td.qvel[tm.jnt_dofadr[jid]]=1
        pid=mujoco.mj_name2id(tm,mujoco.mjtObj.mjOBJ_GEOM,'clearance_probe_geom');bid=mujoco.mj_name2id(tm,mujoco.mjtObj.mjOBJ_BODY,'clearance_probe')
        hits=0
        for _ in range(220):
            mujoco.mj_step(tm,td)
            hits+=sum(pid in (c.geom1,c.geom2) for c in td.contact)
        final_x=float(td.xpos[bid,0]);passage_results[label]={'final_x':final_x,'contact_samples':hits}
        if label=='fits':assert final_x>cx+.4,passage_results
        else:assert final_x<cx-.25 and hits>0,passage_results
    checks['tunnel_physical_pass_and_block']=passage_results
    checks['scene_geometries']=m.ngeom
    checks['status']='passed'
    return checks


def policy_rollout(scene,duration=20.0):
    """Exercise the unmodified controller against real MuJoCo dynamics, no GUI."""
    import sys,runpy,time
    from contextlib import nullcontext
    from types import SimpleNamespace
    source=ROOT/'python_tools'/'onnx_mj_chuanlian.py'
    namespace=runpy.run_path(str(source));g=namespace['main'].__globals__
    state={};started=time.monotonic()
    class HeadlessViewer:
        def __init__(self,model,data,**kwargs):
            self.m=model;self.d=data;self.cam=mujoco.MjvCamera();self.opt=mujoco.MjvOption()
            self.user_scn=SimpleNamespace(flags=np.zeros(int(mujoco.mjtRndFlag.mjNRNDFLAG),dtype=np.uint8))
            self.base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_Link_del')
            self.samples=[];state['viewer']=self
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def lock(self):return nullcontext()
        def is_running(self):return self.d.time<duration
        def sync(self):
            t=float(self.d.time);g['cmd_lin_vel_target']=.3 if 5<t<12 else 0.0
            assert np.isfinite(self.d.qpos).all(),'Non-finite robot state'
            if not self.samples or t-self.samples[-1]['t']>.1:
                self.samples.append(dict(t=t,position=self.d.xpos[self.base].tolist(),upright_cosine=float(self.d.xmat[self.base].reshape(3,3)[2,2]),contacts=int(self.d.ncon)))
    old_launch=mujoco.viewer.launch_passive;old_argv=sys.argv
    try:
        mujoco.viewer.launch_passive=HeadlessViewer
        sys.argv=[str(source),'--xml',str(scene),'--device','cpu','--keyboard-backend','viewer']
        namespace['main']()
    finally:
        mujoco.viewer.launch_passive=old_launch;sys.argv=old_argv
    viewer=state['viewer'];samples=[x for x in viewer.samples if x['t']>=2]
    upright=min(x['upright_cosine'] for x in samples)
    displacement=float(np.linalg.norm(np.array(samples[-1]['position'][:2])-samples[0]['position'][:2]))
    assert upright>.8,'Robot fell in the flat-ground rollout'
    assert displacement>.5,'Robot did not move in response to the command'
    assert not np.any(viewer.d.warning.number),'MuJoCo numerical warning'
    return dict(simulated_seconds=float(viewer.d.time),wall_seconds=time.monotonic()-started,min_upright_cosine=upright,displacement_m=displacement,numerical_warnings=viewer.d.warning.number.tolist(),status='passed',scope='flat ground only; does not certify all obstacles traversable')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene',type=Path,default=ROOT/'scenes/rm2026_arena.xml')
    a.add_argument('--parameters',type=Path,default=ROOT/'scenes/rm2026_parameters.json')
    a.add_argument('--rollout',action='store_true',help='Also run20s of the existing ONNX policy on flat ground')
    a.add_argument('--report',type=Path)
    args=a.parse_args()
    result=verify(args.scene.resolve(),args.parameters.resolve())
    if args.rollout:result['policy_rollout']=policy_rollout(args.scene.resolve())
    text=json.dumps(result,indent=2,ensure_ascii=False)
    print(text)
    if args.report:args.report.write_text(text+'\n')

if __name__=='__main__':main()
