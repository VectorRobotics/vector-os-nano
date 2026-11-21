"""Prepare the pinned ArtVIP home with an accessible storage wing and contact-only cart."""
from pathlib import Path
import argparse, hashlib, json, shutil
import mujoco as mj
import numpy as np
from PIL import Image
from pxr import Usd, UsdPhysics
from prepare_usd import check_source, normalize, write_mjcf
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--cart',type=Path,required=True)
a=parser.parse_args(); source=a.source.resolve(); out=a.output.resolve()
check_source(source,Path(__file__).resolve().parents[1]/'examples/household/sources.json')
out.mkdir(parents=True,exist_ok=True)
prepared=out/'prepared.usdc'
stage=Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
for prim in list(stage.Traverse()):
 if prim.IsA(UsdPhysics.Joint): stage.RemovePrim(prim.GetPath())
 elif prim.HasAPI(UsdPhysics.RigidBodyAPI): prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
normalize(stage).Export(str(prepared))
s=mj.MjSpec.from_file(str(prepared)); s.modelname='artvip_house_background'
s.option.timestep=.002
for tex in s.textures:
 if tex.file:
  path=Path(tex.file)
  if path.suffix.lower()=='.png' and path.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':
   dest=out/(hashlib.sha256(path.read_bytes()).hexdigest()+'.png')
   with Image.open(path) as im: im.save(dest)
   tex.file=str(dest)
for j in list(s.joints): s.delete(j)
meshes={m.name:m for m in s.meshes}
repairs=[]
for g in list(s.geoms):
 g.density=0
 if g.type!=mj.mjtGeom.mjGEOM_MESH: continue
 mesh=meshes[g.meshname];mesh.inertia=mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
 if not (g.contype or g.conaffinity): continue
 g.friction=[.7,.005,.0001]
 points=np.asarray(mesh.uservert).reshape(-1,3)*mesh.scale
 values,axes=np.linalg.eigh(np.cov(points.T))
 if values[0]<1e-10:
  name=g.name+'_panel_collision'
  # 2 mm symmetric surface thickness, only for the source's zero-volume colliders.
  import trimesh
  hull=trimesh.convex.convex_hull(np.concatenate([points+.001*axes[:,0],points-.001*axes[:,0]]))
  s.add_mesh(name=name,uservert=hull.vertices.ravel(),userface=hull.faces.ravel())
  g.parent.add_geom(name=name,type=mj.mjtGeom.mjGEOM_MESH,meshname=name,pos=g.pos,quat=g.quat,
                    group=3,friction=g.friction,density=0)
  g.contype=g.conaffinity=0;repairs.append(g.name)
# Extension: 6.48 x 4 m; original source furniture is not moved or rescaled.
def box(name,pos,size,rgba):
 return s.worldbody.add_geom(name=name,type=mj.mjtGeom.mjGEOM_BOX,pos=pos,size=size,rgba=rgba,
                             density=0,friction=[.7,.005,.0001])
oak=[.57,.42,.27,1];wall=[.88,.86,.81,1]
box('entry_storage_floor',[4.13728,-2.85816,-.05],[3.24,2,.05],oak)
box('entry_left_wall',[.83728,-2.85816,1.5],[.06,2.06,1.5],wall)
box('entry_right_wall',[7.43728,-2.85816,1.5],[.06,2.06,1.5],wall)
box('entry_front_left',[2.05,-4.91816,1.5],[1.21272,.06,1.5],wall)
box('entry_front_right',[5.9,-4.91816,1.5],[1.53728,.06,1.5],wall)
box('entry_door_lintel',[3.975,-4.91816,2.7],[.71228,.06,.3],wall)
# Storage shelving has individual boards and posts, leaving its interior accessible.
for z in (.18,.76,1.22,1.68): box(f'shelf_board_{z}',[1.5,-2.1,z],[.27,.8,.018],oak)
for y in (-2.87,-1.33):
 for x in (1.25,1.75): box(f'shelf_post_{x}_{y}',[x,y,.95],[.022,.022,.95],[.21,.23,.22,1])
s.worldbody.add_light(pos=[4,-2,3],dir=[0,0,-1],diffuse=[.7,.65,.6])
assert not list(s.joints), 'Background must be explicitly fixed'
cart=mj.MjSpec.from_file(str(a.cart.resolve()))
# World placement only: the cart retains its free root, passive drawer and casters.
for geom in list(cart.worldbody.geoms): cart.delete(geom)
for light in list(cart.lights): cart.delete(light)
cart.meshdir=str(a.cart.resolve().parent / (cart.meshdir or ''))
cart.texturedir=str(a.cart.resolve().parent / (cart.texturedir or ''))
s.attach(cart,prefix='cart_',frame=s.worldbody.add_frame(pos=[5.4,-2.1,0],quat=[2**-.5,0,0,-2**-.5]))
objects = [
 ('tissue_refill',[1.5,-2.65,.814],[.025,.08,.035],.18,[.88,.83,.70,1]),
 ('sponge_pack_1',[1.5,-2.3,.799],[.025,.06,.02],.06,[.94,.72,.19,1]),
 ('sponge_pack_2',[1.5,-1.95,.799],[.025,.06,.02],.06,[.28,.58,.38,1]),
 ('zip_bag_box',[1.5,-1.6,.799],[.025,.07,.02],.12,[.29,.52,.75,1])]
for name,pos,size,mass,rgba in objects:
 body=s.worldbody.add_body(name=name,pos=pos);body.add_freejoint(name=name+'_free')
 body.add_geom(name=name,type=mj.mjtGeom.mjGEOM_BOX,size=size,mass=mass,rgba=rgba,friction=[.8,.005,.0001],condim=4)
s.option.timestep=.001;s.option.integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST
s.option.cone=mj.mjtCone.mjCONE_ELLIPTIC;s.option.iterations=100
s.visual.global_.offwidth=1280;s.visual.global_.offheight=960
for g in s.geoms: g.solref=[.004,1];g.solimp=[.95,.99,.001,.5,2]
# Keep original pixels on disk; MJ4 loads them at render time instead of copying them per physics model.
for texture in s.textures:
 if texture.file: texture.embed=mj.mjtEmbed.mjEMBED_FALSE
m=s.compile()
assert m.nq == 44 and m.nu == 0, 'Only the free cart, passive joints and four free packages may move'
shutil.copyfile(Path(__file__).resolve().parents[1]/'examples/household/worlds/artvip_home/world.yaml',out/'world.yaml')
write_mjcf(s, out/'scene.xml')
mj.mj_saveModel(m,str(out/'scene.mjb'))
json.dump(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 dataset_revision='d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e',ngeom=m.ngeom,nmesh=m.nmesh,
 fixed_background=True, cart_source=str(a.cart.resolve()), cart_sha256=hashlib.sha256(a.cart.read_bytes()).hexdigest(),
 object_parameters=[dict(name=n,position=p,half_size=z,mass_kg=m,source='engineering estimate, rigid sealed package') for n,p,z,m,c in objects],
 cart_brakes='initially engaged via rolling frictionloss, no pedal actuation yet',source_furniture_transform_changes=False,planar_panels_2mm=repairs,
 collision_status='Source convex hulls on fixed background only; not validated for contact manipulation. Core drawer and objects use separate cavity-preserving colliders.',
 estimated=['extension dimensions','panel thickness','friction'],physical_verdict='unverified'),(out/'asset-audit.json').open('w'),indent=2)
print('BUILT HOME',m.ngeom,m.nmesh,flush=True)
