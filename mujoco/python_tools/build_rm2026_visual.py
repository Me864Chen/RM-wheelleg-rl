#!/usr/bin/env python3
"""Add reproducible RM2026 visual styling without changing motion-scene collisions.

Reference: rulebook V2.2.0 chapter4, figures4-1/2/5/11/20/21/24.
Decorations are not usable collision surfaces; simplified machinery stays cosmetic.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, struct, zlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
PALETTE={'black':'.045 .049 .052 1','metal':'.24 .26 .27 1','trim':'.52 .55 .55 1',
 'white':'.83 .85 .83 1','red':'.67 .055 .045 1','blue':'.045 .18 .72 1',
 'amber':'.9 .37 .045 1','olive':'.29 .31 .25 1','green':'.08 .8 .24 1'}

def fmt(v):return ' '.join(f'{float(x):.8g}' for x in v)

def png(path,rgb):
    h,w,_=rgb.shape
    def chunk(t,b):return struct.pack('!I',len(b))+t+b+struct.pack('!I',zlib.crc32(t+b)&0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+row.tobytes() for row in rgb),6))+chunk(b'IEND',b''))

def texture(path,base,seed):
    rng=np.random.default_rng(seed);n=512
    # Fine PVC grain and irregular tile patches, no distracting checkerboard.
    fine=rng.normal(0,.7,(n,n));coarse=rng.normal(0,.8,(16,16)).repeat(32,0).repeat(32,1)
    noise=fine+coarse
    arr=np.clip(np.array(base)[None,None,:]+noise[:,:,None],0,255).astype(np.uint8)
    arr[::128,:,:]=np.clip(arr[::128,:,:].astype(int)-5,0,255)
    arr[:,::128,:]=np.clip(arr[:,::128,:].astype(int)-5,0,255)
    png(path,arr)

class Visual:
    def __init__(self,source,output):
        self.source=source;self.output=output;self.tree=ET.parse(source);self.root=self.tree.getroot()
        self.assets=self.root.find('asset');self.world=self.root.find('worldbody');self.count=0
        # Ray queries use the original terrain only. Nothing painted can affect a ray.
        self.m=mujoco.MjModel.from_xml_path(str(source));self.d=mujoco.MjData(self.m);mujoco.mj_forward(self.m,self.d)
        for i in range(self.m.ngeom):
            name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
            self.m.geom_group[i]=0 if name.startswith('rm_') else 5
        self.groups=np.array([1,0,0,0,0,0],dtype=np.uint8)
        for e in self.assets:
            if e.get('file'):e.set('file',os.path.relpath((source.parent/e.get('file')).resolve(),output.parent))
        for key,rgba in PALETTE.items():
            ET.SubElement(self.assets,'material',name='vis_'+key,rgba=rgba,specular='.18',shininess='.25')
        tdir=output.parent/'textures';tdir.mkdir(parents=True,exist_ok=True)
        for name,base in [('pvc',(55,57,58))]:
            file=tdir/(name+'.png');texture(file,base,2026 if name=='pvc' else 2027)
            ET.SubElement(self.assets,'texture',name='vis_'+name,type='2d',file=os.path.relpath(file,output.parent))
        for mesh in self.assets.findall('mesh'):
            if mesh.get('name','').startswith('rm_mesh_floor_') and mesh.get('vertex'):
                vertices=np.array([float(v) for v in mesh.get('vertex').split()]).reshape(-1,3)
                mesh.set('texcoord',fmt(vertices[:,:2].ravel()/4))
        for mat in self.assets.findall('material'):
            name=mat.get('name')
            if name=='rm_floor':mat.attrib.update(texture='vis_pvc',rgba='.8 .8 .8 1',texrepeat='1 1',texuniform='false')
            elif name in ['rm_deck','rm_ramp']:
                mat.attrib.pop('texture',None)
                mat.attrib.update(rgba='.27 .29 .28 1',specular='.03')
            elif name=='rm_rail':mat.attrib.update(rgba='.16 .18 .18 1')
        # Hide tall perimeter faces visually while retaining the exact 2.4m collision boundary.
        # Low opaque boards below and transparent upper safety screens match the render style.
        for e in self.world.findall('geom'):
            name=e.get('name','')
            if name.startswith('rm_wall_'):e.set('rgba','.12 .14 .15 .10')
            if '_base_target_' in name:e.set('rgba','0 0 0 0')
        vis=self.root.find('visual');vis.find('headlight').attrib.update(ambient='.45 .45 .45',diffuse='.65 .65 .65',specular='.1 .1 .1')
        ET.SubElement(vis,'rgba',haze='.1 .12 .14 1')
        ET.SubElement(self.world,'light',name='vis_key_light',pos='0 0 12',dir='0 0 -1',directional='true',diffuse='.55 .55 .55',ambient='.15 .15 .15',castshadow='false')
    def name(self):self.count+=1;return f'vis_{self.count:04d}'
    def geom(self,kind,material,**attrs):
        return ET.SubElement(self.world,'geom',name=self.name(),type=kind,material='vis_'+material,contype='0',conaffinity='0',group='0',**attrs)
    def ground(self,y):return -abs(y)*math.tan(math.radians(1.5))
    def height(self,x,y):
        hit=np.zeros(1,dtype=np.int32)
        dist=mujoco.mj_ray(self.m,self.d,np.array([x+1e-7,y+1e-7,4.]),np.array([0.,0.,-1.]),self.groups,1,-1,hit)
        return 4-dist if dist>=0 else self.ground(y)
    def box(self,center,size,material='metal',yaw=0):
        self.geom('box',material,pos=fmt(center),size=fmt(size),euler=fmt([0,0,yaw]))
    def rod(self,a,b,r,material='metal'):
        self.geom('cylinder',material,fromto=fmt(list(a)+list(b)),size=str(r))
    def polygon(self,poly,z,material):
        # Explicit coplanar triangle faces for paint; no convex collision shape.
        top=[(x,y,z(x,y) if callable(z) else z) for x,y in poly]
        n=len(top);verts=top+[(x,y,h-.0005) for x,y,h in top]
        faces=[(0,i,i+1) for i in range(1,n-1)]+[(n,n+i+1,n+i) for i in range(1,n-1)]
        for i in range(n):
            j=(i+1)%n;faces.extend([(i,n+i,n+j),(i,n+j,j)])
        mesh=self.name()+'_mesh'
        ET.SubElement(self.assets,'mesh',name=mesh,vertex=fmt([v for p in verts for v in p]),face=' '.join(str(v) for f in faces for v in f))
        self.geom('mesh',material,mesh=mesh)
    def line(self,points,material='white',width=.035,closed=False):
        points=list(points)
        if closed:points.append(points[0])
        for a,b in zip(points,points[1:]):
            dx=b[0]-a[0];dy=b[1]-a[1];length=math.hypot(dx,dy)
            if length<1e-5:continue
            # Follow raised/sloping surfaces; split at elevation discontinuities.
            n=max(1,math.ceil(length/.18));ox=-dy/length*width/2;oy=dx/length*width/2
            for j in range(n):
                u=j/n;v=(j+1)/n
                x0=a[0]+dx*u;y0=a[1]+dy*u;x1=a[0]+dx*v;y1=a[1]+dy*v
                z0=self.height(x0,y0);z1=self.height(x1,y1)
                if abs(z1-z0)>.09:continue
                z=lambda x,y,x0=x0,y0=y0,z0=z0,z1=z1: z0+(z1-z0)*((x-x0)*dx+(y-y0)*dy)/(length*length/n)+.003
                self.polygon([(x0+ox,y0+oy),(x0-ox,y0-oy),(x1-ox,y1-oy),(x1+ox,y1+oy)],z,material)
    def ring(self,c,r,material='white',n=6,width=.035,angle=0):
        self.line([(c[0]+r*math.cos(angle+i*2*math.pi/n),c[1]+r*math.sin(angle+i*2*math.pi/n)) for i in range(n)],material,width,True)
    def mirror(self,points,s):return [(s*x,s*y) for x,y in points]
    def paint(self):
        # Plan approximations from Fig4-1/5, projected onto the existing collision layout.
        for s,color in [(1,'red'),(-1,'blue')]:
            M=lambda p:self.mirror(p,s)
            self.line(M([(-13.7,3.7),(-11,3.7),(-10.3,3.25),(-8.4,3.25),(-6.5,3.25),(-4.2,6.55),(-2.3,2.4),(-2.3,-.6),(-3.7,-.6),(-4.7,-3.6),(-6.5,-3.6),(-7.3,-1.9),(-9,-1.9)]),color,.055)
            self.line(M([(-13.7,-.12),(-12.2,-.12)]),color,.06)
            self.line(M([(-13.7,.12),(-12.2,.12)]),color,.06)
            self.ring((-7.6*s,0),.57,color,6,.045)
            self.ring((-11.6*s,0),1.22,'white',6,.035,math.pi/6)
            self.ring((-7.6*s,0),1.3,'white',6,.025)
            # Landing pad in the outer rear corner (paint only, no invented step).
            self.ring((-12.4*s,5.75*s),.95,'trim',8,.10,math.pi/8)
            self.ring((-12.4*s,5.75*s),.79,color,8,.05,math.pi/8)
            self.line(M([(-13.4,-4.1),(-11.5,-4.1),(-11.5,-6.9),(-13.4,-6.9),(-13.4,-4.1)]),'white',.035)
            self.line(M([(-11.1,-7),(-11.1,-4.3),(-10.9,-4.05)]),'white',.045)
            # R-like stencil, simple geometric emblem rather than an external logo texture.
            x,y=-10.5,3.65
            self.line(M([(x-.2,y-.3),(x-.2,y+.3),(x+.15,y+.3),(x+.23,y+.17),(x+.15,y),(x-.2,y),(x+.25,y-.3)]),'white',.07)
            self.line(M([(-11.05,3.1),(-9.8,3.1),(-9.8,4.2),(-11.05,4.2)]),color,.05,True)
            for x in [-4.0,-3.65,-3.3]:
                self.line(M([(x-.12,6.75),(x+.08,6.6),(x-.12,6.45)]),'white',.035)
            # Lane lines and warning stripes remain outside the actual fly-ramp gap.
            self.line(M([(2.52,5.91),(7.6,5.91)]),'white',.025)
            self.line(M([(2.52,6.77),(7.6,6.77)]),'white',.025)
            for j in range(8):
                x=2.55+j*.18
                self.line(M([(x,6.84),(x+.14,7.04)]),'amber',.05)
            for x in [-.65,.1]:self.line(M([(x,5.96),(x,6.72)]),'white',.045)
        p=json.loads(self.source.with_name('rm2026_parameters.json').read_text())
        central=p['central']['footprint']
        self.line([(x*.93,y*.96) for x,y in central],'white',.045,True)
        self.line([(-1.0,1.35),(-1.7,.65),(-1.7,-.65),(-.6,-1.35)],'red',.045)
        self.line([(1.0,-1.35),(1.7,-.65),(1.7,.65),(.6,1.35)],'blue',.045)
    def infrastructure(self):
        # Low perimeter kickboards + a light mesh fence. Collision stays in original walls.
        for x in [-14,14]:
            self.box((x,0,-.02),(.035,7.5,.14),'black')
        for y in [-7.5,7.5]:
            self.box((0,y,self.ground(y)+.16),(14,.035,.16),'black')
            for x in np.arange(-14,14.1,2):
                self.rod((x,y,self.ground(y)),(x,y,self.ground(y)+2.4),.012,'black')
            for h in [.6,1.2,1.8,2.4]:
                self.rod((-14,y,self.ground(y)+h),(14,y,self.ground(y)+h),.004,'black')
        # Capped concrete/metal edges accent the silhouette without changing contact.
        for s,color in [(1,'red'),(-1,'blue')]:
            self.line(self.mirror([(-10.75,7.19),(-.4,7.19)],s),'trim',.10)
            bx=-11.6*s;g=self.ground(0)
            # Tapered base armour around the original .5m wide, 1.2m high collision envelope.
            for i in range(6):
                a=i*math.pi/3;b=(i+1)*math.pi/3
                vertices=[(bx+r*math.cos(q),r*math.sin(q),g+z) for r,z in [(.32,.16),(.18,1.05)] for q in [a,b]]
                name=self.name()+'_armour'
                vertices=vertices+[(x-.002*math.cos((a+b)/2),y-.002*math.sin((a+b)/2),z) for x,y,z in vertices]
                ET.SubElement(self.assets,'mesh',name=name,vertex=fmt([v for p in vertices for v in p]),face='0 1 3 0 3 2 4 6 7 4 7 5 0 4 5 0 5 1 2 3 7 2 7 6 0 2 6 0 6 4 1 5 7 1 7 3')
                self.geom('mesh','metal',mesh=name)
                self.rod((bx+.29*math.cos(a),.29*math.sin(a),.22),(bx+.17*math.cos(a),.17*math.sin(a),1.03),.017,color)
            self.box((bx,0,1.12),(.22,.15,.065),'black')
            self.box((bx+.225*s,0,1.12),(.01,.06,.025),'green')
            # Resource station skin around the existing supply collision envelope.
            cx=-13.3*s;cy=-5.8*s;gz=self.ground(cy)
            self.box((cx,cy,gz+.13),(.37,.82,.13),'olive')
            self.box((cx-.25*s,cy,gz+.65),(.08,.74,.45),'olive')
            for dy in [-.69,.69]:
                self.box((cx,cy+dy,gz+.65),(.31,.05,.45),'olive')
                self.box((cx+.32*s,cy+dy,gz+.72),(.015,.025,.34),color)
            self.rod((cx,cy-.35,gz+.85),(cx+.15*s,cy+.05,gz+1.0),.065,'black')
            self.rod((cx+.15*s,cy+.05,gz+1.0),(cx+.2*s,cy+.3,gz+.55),.045,'metal')
        # Static central energy-machine silhouette. Existing support collisions retained;
        # overhead rings/blades are decorative, no articulated competition mechanism.
        for y in [-.775,.775]:
            self.box((0,y,1.5),(.13,.06,1.15),'black')
            self.box((.14,y,1.5),(.012,.035,1.08),'trim')
        self.rod((0,-.84,2.55),(0,.84,2.55),.075,'metal')
        # Ring in the YZ plane, above the 300mm assembly island.
        center=np.array([0.,0.,1.65]);rad=.65
        for i in range(24):
            a=i*2*math.pi/24;b=(i+1)*2*math.pi/24
            self.rod(center+np.array([0,rad*math.cos(a),rad*math.sin(a)]),center+np.array([0,rad*math.cos(b),rad*math.sin(b)]),.025,'metal')
        for i in range(5):
            a=i*2*math.pi/5
            self.rod(center,center+np.array([0,.54*math.cos(a),.54*math.sin(a)]),.035,'black')
            self.box((.02,.55*math.cos(a),1.65+.55*math.sin(a)),(.045,.1,.075),'amber')
    def save(self):
        ET.indent(self.root,space='  ');self.tree.write(self.output,encoding='utf-8',xml_declaration=True)
        return self.output

def build(source,output):
    v=Visual(source.resolve(),output.resolve());v.paint();v.infrastructure();v.save()
    manifest={'scene':output.name,'motion_source':source.name,'motion_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'decorative_geometries':sum(g.get('name','').startswith('vis_') for g in v.world.findall('geom')),
      'reference':'RoboMaster 2026 V2.2.0 chapter4 figures4-1/2/5/11/20/21/24',
      'scope':'Approximate visual reconstruction; layout inherited from motion prototype, NOT full CAD or calibrated camera dataset.',
      'unchanged':'Original collision geometry, robot dynamics, timestep, contact parameters, cameras and policy.',
      'approximations':['Line placement traced approximately from rulebook diagrams, not dimensioned survey.','Procedural PVC/concrete appearance, estimated colours under neutral light.','Base/resource station/overhead central energy mechanism simplified, static visual shells; no game mechanics.','Added shells, paint and fence do not collide; original simplified facility envelopes remain the collision boundaries.','2.4m collision perimeter rendered as low opaque kickboard + transparent upper screens.']}
    output.with_name('rm2026_visual_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Generated {output}')
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'scenes/rm2026_arena.xml');p.add_argument('--output',type=Path,default=ROOT/'scenes/rm2026_visual.xml')
    a=p.parse_args();build(a.source,a.output)
if __name__=='__main__':main()
