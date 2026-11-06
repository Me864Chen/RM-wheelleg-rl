#!/usr/bin/env python3
"""Generate separate RM2026 motion scenes; original robot XML is read-only.

Dimension provenance and explicit approximations: scenes/rm2026_parameters.json.
Static terrain is split into convex prisms: holes/tunnels are not filled by hulls.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml')


def fmt(values):
    return ' '.join(f'{float(x):.9g}' for x in values)


def rect(x0,y0,x1,y1):
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]


def cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def triangulate(poly):
    p=list(map(tuple,poly))
    area=sum(p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p)))
    if area<0:p.reverse()
    result=[];indices=list(range(len(p)))
    while len(indices)>3:
        for j,b in enumerate(indices):
            a=indices[j-1];c=indices[(j+1)%len(indices)]
            if cross(p[a],p[b],p[c])<=1e-10:continue
            if any(all(cross(u,v,p[k])>=-1e-10 for u,v in [(p[a],p[b]),(p[b],p[c]),(p[c],p[a])]) for k in indices if k not in (a,b,c)):continue
            result.append([p[a],p[b],p[c]]);indices.pop(j);break
        else:raise ValueError('Non-simple polygon or degenerate triangulation')
    result.append([p[i] for i in indices]);return result


def split_at_ridge(poly):
    """Split at y=0 so the floor's two slopes are not bridged by a convex hull."""
    if min(y for x,y in poly)>=-1e-10 or max(y for x,y in poly)<=1e-10:
        return [poly]
    pieces=[]
    for sign in (1,-1):
        out=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            ina=sign*a[1]>=0;inb=sign*b[1]>=0
            if ina:out.append(a)
            if ina!=inb:
                t=-a[1]/(b[1]-a[1]);out.append((a[0]+t*(b[0]-a[0]),0.0))
        clean=[]
        for v in out:
            if not clean or abs(v[0]-clean[-1][0])+abs(v[1]-clean[-1][1])>1e-9:clean.append(v)
        if len(clean)>1 and abs(clean[0][0]-clean[-1][0])+abs(clean[0][1]-clean[-1][1])<1e-9:clean.pop()
        if len(clean)>=3:pieces.append(clean)
    return pieces


class Arena:
    def __init__(self,source,params,output_dir,reference_dir=None):
        self.params=params;self.root=ET.parse(source).getroot();self.output_dir=output_dir
        self.assets=self.root.find('asset');self.world=self.root.find('worldbody')
        self.geom_names=[];self.features=[];self.materials={}
        for child in list(self.world):
            if child.tag!='body':self.world.remove(child)
        # Resolve source resources before rebasing: never copy large robot meshes.
        reference_dir=reference_dir or output_dir
        for e in self.assets:
            if e.get('file'):
                absolute=(source.parent/e.get('file')).resolve()
                e.set('file',os.path.relpath(absolute,reference_dir))
        for e in list(self.assets):
            if e.tag=='hfield':self.assets.remove(e)
        opt=self.root.find('option');opt.set('timestep','0.002');opt.set('iterations','100')
        # Explicitly use the 5ms dynamics step assumed by the existing policy script.
        visual=ET.SubElement(self.root,'visual')
        ET.SubElement(visual,'global',offwidth='1280',offheight='960')
        ET.SubElement(visual,'headlight',ambient='0.45 0.45 0.45',diffuse='0.6 0.6 0.6',specular='0.1 0.1 0.1')
        ET.SubElement(self.assets,'texture',name='rm_floor_tex',type='2d',builtin='checker',width='256',height='256',rgb1='.18 .19 .2',rgb2='.21 .22 .23')
        for name,rgba in [('floor','.22 .23 .24 1'),('deck','.29 .3 .31 1'),('ramp','.38 .39 .4 1'),('rail','.56 .55 .49 1'),('red','.55 .07 .07 1'),('blue','.06 .16 .6 1'),('rough','.12 .12 .13 1')]:
            attrs=dict(name='rm_'+name,rgba=rgba,specular='.05',reflectance='0')
            if name=='floor':attrs.update(texture='rm_floor_tex',texrepeat='14 7.5',texuniform='true')
            ET.SubElement(self.assets,'material',**attrs)
        default=self.root.find('default');terrain=ET.SubElement(default,'default',{'class':'rm_terrain'})
        contact=params['contact']
        ET.SubElement(terrain,'geom',contype='2',conaffinity='1',condim='3',priority='1',friction=fmt([contact['sliding_friction'],contact['torsional_friction'],contact['rolling_friction']]),solref=fmt(contact['solref']),solimp='.95 .99 .001',margin='0',gap='0',group='0')

    def ground(self,y):return -abs(y)*math.tan(math.radians(self.params['field']['crossfall_deg']))

    def mesh(self,name,vertices,material='deck'):
        meshname='rm_mesh_'+name
        ET.SubElement(self.assets,'mesh',name=meshname,vertex=fmt([v for p in vertices for v in p]))
        ET.SubElement(self.world,'geom',name='rm_'+name,type='mesh',mesh=meshname,material='rm_'+material,**{'class':'rm_terrain'})
        self.geom_names.append('rm_'+name)

    def slab(self,name,poly,top,bottom=None,material='deck'):
        """Top/bottom in local floor-height units; top can be planar function."""
        gettop=top if callable(top) else lambda x,y:top
        getbottom=bottom if callable(bottom) else (lambda x,y:bottom) if bottom is not None else lambda x,y:-.3
        for k,piece in enumerate(split_at_ridge(list(poly))):
            for j,tri in enumerate(triangulate(piece)):
                vs=[(x,y,self.ground(y)+getbottom(x,y)) for x,y in tri]+[(x,y,self.ground(y)+gettop(x,y)) for x,y in tri]
                if max(v[2] for v in vs)-min(v[2] for v in vs)<1e-7:raise ValueError(name)
                self.mesh(f'{name}_{k}_{j}',vs,material)

    def box(self,name,x0,y0,x1,y1,height,bottom=None,material='deck'):
        self.slab(name,rect(x0,y0,x1,y1),height,bottom,material)

    def ramp(self,name,origin,length,width,start,end,heading=0,material='ramp'):
        cx,cy=origin;c=math.cos(heading);s=math.sin(heading)
        def trans(x,y):return (cx+c*x-s*y,cy+s*x+c*y)
        poly=[trans(x,y) for x,y in rect(0,-width/2,length,width/2)]
        self.slab(name,poly,lambda x,y:start+(end-start)*((x-cx)*c+(y-cy)*s)/length,material=material)
        self.features.append(dict(name=name,kind='ramp',origin=list(origin),heading=heading,length=length,width=width,start=start,end=end,slope_deg=math.degrees(math.atan2(abs(end-start),length))))

    def mirrored_slab(self,name,poly,height,material='deck'):
        self.slab('red_'+name,poly,height,material=material)
        self.slab('blue_'+name,[(-x,-y) for x,y in poly],height,material=material)

    def add_floor(self):
        f=self.params['field'];L=f['length']/2;W=f['width']/2
        self.box('floor_north',-L,0,L,W,0,bottom=-.2,material='floor')
        self.box('floor_south',-L,-W,L,0,0,bottom=-.2,material='floor')
        t=f['wall_thickness'];h=f['wall_top_above_floor']
        for name,x0,y0,x1,y1 in [('west',-L-t,-W-t,-L,W+t),('east',L,-W-t,L+t,W+t),('north',-L,W,L,W+t),('south',-L,-W-t,L,-W)]:
            self.box('wall_'+name,x0,y0,x1,y1,h,material='rail')

    def add_trapezoids(self):
        p=self.params['trapezoid'];ox,oy=p['northwest_origin']
        poly=[(ox+x,oy+y) for x,y in p['outline']]
        self.mirrored_slab('trapezoid',poly,p['deck_height'])
        # Separate 400mm pad and access wedges; shape subdivision is documented.
        high=rect(ox+.15,oy-4.38,ox+1.85,oy-2.922)
        self.mirrored_slab('high_pad',high,p['high_deck_height'])
        for side in [1,-1]:
            rise=p['high_deck_height'];run=rise/math.tan(math.radians(p['steep_deg']))
            self.ramp(('red' if side==1 else 'blue')+'_43deg',(side*(ox+1.0),side*(oy-4.38-run)),run,1.7,0,rise,math.pi/2 if side==1 else -math.pi/2)
            rise=p['high_deck_height']-p['deck_height'];run=rise/math.tan(math.radians(p['shallow_deg']))
            self.ramp(('red' if side==1 else 'blue')+'_23deg',(side*(ox+1.85),side*(oy-3.65)),run,1.0,p['high_deck_height'],p['deck_height'],0 if side==1 else math.pi)
        run=(p['high_deck_height']-p['deck_height'])/math.tan(math.radians(p['shallow_deg']))
        self.mirrored_slab('high_pad_link',rect(ox+1.85+run,oy-4.15,ox+2.85,oy-3.15),p['deck_height'])
        # Edge guards only along the outer north/south border; no invisible access walls.
        self.mirrored_slab('highland_guard',rect(ox,oy-.15,ox+10.505,oy),.35,'rail')

    def add_central(self):
        p=self.params['central'];self.slab('central_deck',p['footprint'],p['deck_height'])
        rise=p['deck_height'];run=rise/math.tan(math.radians(p['entry_deg']))
        for name,origin,heading in [('north',(-1.15,4.9695+run),-math.pi/2),('south',(1.15,-4.9695-run),math.pi/2)]:
            self.ramp('central_'+name+'_10_5deg',origin,run,1.0,0,rise,heading)
        # Simplified 300mm assembly island above a 200mm deck.
        halfx=p['inner_top_width']/2;halfy=p['inner_top_length']/2
        self.box('assembly_top',-halfx,-halfy,halfx,halfy,p['inner_height'])
        delta=p['inner_height']-p['deck_height']
        for name,origin,width,angle,heading in [('east',(halfx,0),2*halfy,12,0),('west',(-halfx,0),2*halfy,14,math.pi),('north',(0,halfy),2*halfx,15,math.pi/2),('south',(0,-halfy),2*halfx,45,-math.pi/2)]:
            self.ramp('assembly_'+name+'_'+str(angle)+'deg',origin,delta/math.tan(math.radians(angle)),width,p['inner_height'],p['deck_height'],heading)
        # Static non-driving machinery envelope, kept above the driveable assembly slopes.
        self.box('energy_support_left',-.18,-.85,.18,-.7,2.7,bottom=.3,material='rail')
        self.box('energy_support_right',-.18,.7,.18,.85,2.7,bottom=.3,material='rail')

    def add_road_fly_tunnel(self):
        p=self.params['fly'];t=self.params['tunnel'];road=self.params['road'];w=p['outer_width']
        x,y=p['red_lip_center'];run=(p['lip_height']-p['tail_height'])/math.tan(math.radians(p['slope_deg']))
        for side,label in [(1,'red'),(-1,'blue')]:
            transform=lambda px,py:(side*px,side*py)
            # No elevated floor spans the 650mm gap; only the arena floor remains below.
            poly=rect(-.295-t['length']/2,y-w/2,x-p['gap'],y+w/2)
            self.slab(label+'_road_landing',[transform(*q) for q in poly],p['landing_height'])
            self.ramp(label+'_fly_17deg',transform(x,y),run,p['lane_width'],p['lip_height'],p['tail_height'],0 if side==1 else math.pi)
            # The side lips lie outside the 860mm usable ramp corridor.
            for i,dy in enumerate([-(w+p['lane_width'])/4,(w+p['lane_width'])/4]):
                self.ramp(label+'_fly_guard_'+str(i),transform(x,y+dy),run,(w-p['lane_width'])/2,p['lip_height']+.10,p['tail_height']+.10,0 if side==1 else math.pi,material='rail')
            self.slab(label+'_road_tail',[transform(*q) for q in rect(x+run,y-w/2,8.0,y+w/2)],p['tail_height'])
            # Road access: exact 11/15 degree inclines, simplified locations.
            for i,angle in enumerate(road['slopes_deg']):
                length=road['deck_height']/math.tan(math.radians(angle))
                self.ramp(label+'_road_access_'+str(angle),transform(7.7 if i==0 else -.295,y-w/2-length),length,.8,0,road['deck_height'],math.pi/2 if side==1 else -math.pi/2)
            cx,cy=t['red_center'];L=t['length'];inner=t['inner_width'];outer=t['outer_width'];z=road['deck_height'];h=t['clear_height']
            for j,(a,b) in enumerate([(-outer/2,-inner/2),(inner/2,outer/2)]):
                self.slab(label+'_tunnel_side_'+str(j),[transform(*q) for q in rect(cx-L/2,cy+a,cx+L/2,cy+b)],z+h+t['roof_thickness'],bottom=z,material='rail')
            self.slab(label+'_tunnel_roof',[transform(*q) for q in rect(cx-L/2,cy-outer/2,cx+L/2,cy+outer/2)],z+h+t['roof_thickness'],bottom=z+h,material='rail')
            # Exposed vertical road edge of 250mm, separate from tunnel floor.
            self.slab(label+'_road_250mm',[transform(*q) for q in rect(8.0,y-w/2,9.0,y+w/2)],.25)
        self.features.extend([dict(name='fly',kind='gap',lip_x=x,landing_edge_x=x-p['gap'],y=y,gap=p['gap'],width=p['lane_width'],ground_below=True),dict(name='tunnel',kind='clearance',center=t['red_center'],inner_width=t['inner_width'],height=t['clear_height'],floor=road['deck_height'])])

    def add_rough(self):
        p=self.params['rough'];cx,cy=p['red_center'];x0=cx-p['length']/2;w=p['width'];pitch=p['crest_pitch'];base=self.params['trapezoid']['deck_height']
        steps=math.ceil(p['length']/pitch*p['segments_per_pitch'])
        xs=[x0+p['length']*i/steps for i in range(steps+1)]
        def height(x):return base+p['base_thickness']+(p['height_envelope']-p['base_thickness'])*(.5+.5*math.cos(2*math.pi*(x-x0)/pitch))
        for label,s in [('red',1),('blue',-1)]:
            for i,(a,b) in enumerate(zip(xs,xs[1:])):
                # Linear segments approximate the documented-but-unspecified bump profile.
                self.ramp(f'{label}_rough_{i}',(s*a,s*cy),b-a,w,height(a),height(b),0 if s==1 else math.pi,material='rough')
        self.features.append(dict(name='rough',kind='profile',nominal_envelope=p['height_envelope'],pitch=pitch,status='interpretation_pending_CAD'))

    def add_bases_fortresses(self):
        p=self.params['fortress'];radius=p['outer_edge'];inner=p['inner_edge']
        for side,label in [(1,'red'),(-1,'blue')]:
            cx,cy=p['red_center'];cx*=side;cy*=side
            poly=[(cx+inner*math.cos(i*math.pi/3),cy+inner*math.sin(i*math.pi/3)) for i in range(6)]
            self.slab(label+'_fortress_top',poly,p['height'])
            # Preserve all dimensioned sizes; the derived angle is within nominal20deg tolerance.
            apothem=inner*math.cos(math.pi/6);outer_apothem=radius*math.cos(math.pi/6)
            top_r=apothem/math.cos(math.pi/6);bottom_r=outer_apothem/math.cos(math.pi/6)
            for i in range(6):
                a=i*math.pi/3;b=(i+1)*math.pi/3
                top=[(cx+top_r*math.cos(q),cy+top_r*math.sin(q)) for q in (a,b)]
                bottom=[(cx+bottom_r*math.cos(q),cy+bottom_r*math.sin(q)) for q in (a,b)]
                n=(math.cos((a+b)/2),math.sin((a+b)/2))
                self.slab(f'{label}_fortress_slope_{i}',[top[0],bottom[0],bottom[1],top[1]],lambda x,y,cx=cx,cy=cy,n=n:p['height']-((x-cx)*n[0]+(y-cy)*n[1]-apothem)*p['height']/(outer_apothem-apothem))
            # Base envelopes are intentionally simple: their positions are layout estimates.
            bx=-11.6*side;by=0
            self.slab(label+'_base_plinth',[(bx+.65*math.cos(i*math.pi/3),by+.65*math.sin(i*math.pi/3)) for i in range(6)],.15,material='rail')
            self.box(label+'_base_target',bx-.25,-.25,bx+.25,.25,1.2,bottom=.15,material=label)
            self.box(label+'_supply',side*-13.3-.35,side*-5.8-.8,side*-13.3+.35,side*-5.8+.8,1.1,material='rail')
        self.features.append(dict(name='fortress',kind='approximation',outer_edge=radius,inner_edge=inner,height=p['height'],derived_slope_deg=math.degrees(math.atan(p['height']/(outer_apothem-apothem)))))

    def spawn(self):
        p=self.params['spawn'];x,y=p['xy'];body=self.world.find("body[@name='base_Link_del']")
        body.set('pos',fmt([x,y,self.ground(y)+p['body_height_above_floor']]))
        yaw=math.radians(p['yaw_deg']);body.set('quat',fmt([math.cos(yaw/2),0,0,math.sin(yaw/2)]))

    def save(self,name):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        ET.indent(self.root,space='  ')
        path=self.output_dir/name
        ET.ElementTree(self.root).write(path,encoding='utf-8',xml_declaration=True)
        return path


def build(project_root,output_dir,params_path=None,reference_dir=None):
    params=json.loads((params_path or project_root/'scenes/rm2026_parameters.json').read_text())
    arena=Arena(project_root/SOURCE,params,output_dir,reference_dir)
    arena.add_floor();arena.add_trapezoids();arena.add_central();arena.add_road_fly_tunnel();arena.add_rough();arena.add_bases_fortresses();arena.spawn()
    path=arena.save('rm2026_arena.xml')
    manifest=dict(scene=path.name,robot_source=str(SOURCE),robot_source_sha256=hashlib.sha256((project_root/SOURCE).read_bytes()).hexdigest(),geometry_count=len(arena.geom_names),features=arena.features,provenance=params['provenance'])
    (output_dir/'rm2026_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Generated {path} ({len(arena.geom_names)} terrain geoms)')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT,help='MuJoCo directory containing assert_now/')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--parameters',type=Path)
    p.add_argument('--reference-dir',type=Path,help='Resource reference directory for staging tests')
    a=p.parse_args();build(a.project_root,a.output_dir or a.project_root/'scenes',a.parameters,a.reference_dir)

if __name__=='__main__':main()
