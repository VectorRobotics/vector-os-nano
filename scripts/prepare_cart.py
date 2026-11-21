"""2026 ArtVIP IKEA cabinet adapted into a wheeled, braked storage cart; local development asset."""
from pathlib import Path
import argparse,hashlib,json,time
from prepare_usd import check_source, normalize, write_mjcf
import mujoco as mj
import numpy as np
from collision_parts import collision_parts
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--cache',type=Path,required=True)
a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
check_source(a.source,Path(__file__).resolve().parents[1]/'examples/household/sources.json')
from pxr import Usd
normalize(Usd.Stage.Open(Usd.Stage.Open(str(a.source.resolve())).Flatten())).Export(str(out/'prepared.usdc'))
s=mj.MjSpec.from_file(str(out/'prepared.usdc'))
s.modelname='gullaberg_cart';s.option.timestep=.002;s.option.integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST
s.option.cone=mj.mjtCone.mjCONE_ELLIPTIC;s.option.iterations=100
s.option.enableflags |= int(mj.mjtEnableBit.mjENBL_SPECULATIVE)
print('BODIES',[(b.name,b.parent.name if b.parent else None) for b in s.bodies],flush=True)
base=next(b for b in s.bodies if 'E_GULLABERG' in b.name)
drawer=next(b for b in s.bodies if b.name.endswith('E_drawer_4'))
for j in list(s.joints):
 if j.type==mj.mjtJoint.mjJNT_FREE: s.delete(j)
 else: j.damping=[5,0,0];j.frictionloss=1.2
base.add_freejoint(name='cart_free');base.pos=np.asarray(base.pos)+[0,0,.085]
meshes={m.name:m for m in s.meshes};parts_log=[]
for g in list(s.geoms):
 g.density=0
 if g.type!=mj.mjtGeom.mjGEOM_MESH:continue
 mesh=meshes[g.meshname];mesh.inertia=mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
 t=time.monotonic();parts=collision_parts(np.asarray(mesh.uservert).reshape(-1,3)*mesh.scale,np.asarray(mesh.userface).reshape(-1,3),tolerance=.001,surface_thickness=.0002,cache=a.cache)
 g.contype=g.conaffinity=0
 for i,(vertices,faces) in enumerate(parts):
  name=g.name+f'_convex_{i}';s.add_mesh(name=name,uservert=vertices.ravel(),userface=faces.ravel())
  g.parent.add_geom(name=name,type=mj.mjtGeom.mjGEOM_MESH,meshname=name,pos=g.pos,quat=g.quat,
                    density=0,group=3,friction=[.6,.005,.0001])
 parts_log.append(dict(mesh=mesh.name,parts=len(parts),seconds=time.monotonic()-t));print(parts_log[-1],flush=True)
# Engineering mass/inertia estimates, not inferred from possibly overlapping visual shells.
for body,mass,size,com in ((base,14.,np.array([.526,.441,.69]),[0,0,.35]),(drawer,2.6,np.array([.408,.372,.176]),[0,.22,0])):
 body.explicitinertial=True;body.mass=mass;body.ipos=com;body.inertia=mass/12*(np.dot(size,size)-size**2)
for i,(x,y) in enumerate(((-.229966,-.179509),(-.229966,.195325),(.229645,-.179509),(.229813,.195494))):
 swivel=base.add_body(name=f'caster_{i}',pos=[x,y,-.025]);swivel.add_joint(name=f'caster_{i}_swivel',type=mj.mjtJoint.mjJNT_HINGE,axis=[0,0,1],damping=[.05,0,0],frictionloss=.04)
 swivel.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER,size=[.012,.012,0],mass=.12,rgba=[.16,.17,.18,1])
 swivel.add_geom(name=f'caster_{i}_stem',type=mj.mjtGeom.mjGEOM_CYLINDER,pos=[0,0,.0185],size=[.004,.0065,0],density=0,rgba=[.16,.17,.18,1])
 for side in (-1,1):
  swivel.add_geom(name=f'caster_{i}_fork_{side}',type=mj.mjtGeom.mjGEOM_BOX,pos=[.01,side*.011,-.022],size=[.014,.002,.022],density=0,rgba=[.16,.17,.18,1])
 wheel=swivel.add_body(name=f'caster_{i}_wheel',pos=[.02,0,-.035])
 wheel.add_joint(name=f'caster_{i}_roll',type=mj.mjtJoint.mjJNT_HINGE,axis=[0,1,0],damping=[.002,0,0],frictionloss=.08)
 wheel.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER,size=[.025,.008,0],quat=[.7071067811865476,.7071067811865476,0,0],mass=.1,friction=[.8,.005,.0001],rgba=[.09,.095,.10,1])
for name,pos,size in (('divider_x',[0,.22,-.024],[.003,.16,.052]),('divider_y',[0,.22,-.024],[.185,.003,.052])):
 drawer.add_geom(name=name,type=mj.mjtGeom.mjGEOM_BOX,pos=pos,size=size,density=0,rgba=[.73,.70,.64,1],friction=[.6,.005,.0001])
s.worldbody.add_geom(name='development_floor',type=mj.mjtGeom.mjGEOM_PLANE,size=[3,3,.1],friction=[.8,.005,.0001])
s.worldbody.add_light(pos=[1,-2,3],dir=[-1,2,-3])
for g in s.geoms:
 g.solref=[.004,1];g.solimp=[.95,.99,.001,.5,2]
write_mjcf(s,out/'cart.xml');m=s.compile();mj.mj_saveModel(m,str(out/'cart.mjb'))
d=mj.MjData(m);mj.mj_forward(m,d)
report=dict(source=str(a.source.resolve()),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
 dataset_revision='d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e',visual_scale=1.,source_cabinet_height_m=.69,added_caster_mount_height_m=.085,
 rigid_root_free=True,brakes_initially_engaged=True,brake_model='0.08 Nm wheel bearing friction plus 0.04 Nm swivel friction; engineering estimate, pedal actuation not implemented',
 estimated=dict(cabinet_mass_kg=14.,drawer_mass_with_dividers_kg=2.6,uniform_box_inertia=True,drawer_friction_N=1.2,drawer_damping_N_s_m=5.,collider_surface_thickness_m=.0002,contact_solref=[.004,1],contact_solimp=[.95,.99,.001,.5,2]),
 collision_parts=parts_log,initial_max_penetration_m=max(0.,-min((c.dist for c in d.contact),default=0)),physical_verdict='unverified')
(out/'cart-audit.json').write_text(json.dumps(report,indent=2));print('BUILT',m.ngeom,m.nq,report['initial_max_penetration_m'],flush=True)
