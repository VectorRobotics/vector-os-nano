"""Convert the pinned local Malo CAD without publishing it; preserve wheel contacts and URDF limits.

Run in the isolated asset-conversion environment (MuJoCo 4.0.1, trimesh, CoACD).
Output is a user embodiment folder, beside the other folders in VECTOR_HOME/embodiments.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import time
import argparse
import shutil

import mujoco as mj
import numpy as np
import trimesh
from collision_parts import collision_parts

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--cache', type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e6893f32be6cf185e697f59f4d0e468f05061c4d4114d6b774fa8412f778000c'
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
root = ET.parse(source).getroot()
for mesh in root.iter('mesh'):
    mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
node = root.find('mujoco')
if node is None:
    node = ET.SubElement(root,'mujoco')
ET.SubElement(node,'compiler',fusestatic='false',discardvisual='false',inertiafromgeom='false')
s = mj.MjSpec.from_string(ET.tostring(root,encoding='unicode'))
s.option.timestep = .001
s.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
s.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
s.option.iterations = 100
s.body('base_link').add_freejoint(name='base_free')
s.body('base_link').pos = [0,0,0]
meshes = {m.name:m for m in s.meshes}
rows = []
for g in list(s.geoms):
    if not (g.contype or g.conaffinity):
        continue
    g.friction = [.8,.005,.0001]
    g.condim = 4
    g.group = 3
    if g.parent.name != 'base_link' and 'finger' not in g.parent.name and not g.parent.name.endswith(('arm_link_5','arm_link_7')):
        continue
    mesh = meshes[g.meshname]
    geometry = trimesh.load_mesh(mesh.file,process=False)
    t = time.monotonic()
    if g.parent.name == 'base_link':
        parts = collision_parts(np.asarray(geometry.vertices)*mesh.scale,geometry.faces,
                                tolerance=.001,surface_thickness=.0001,cache=args.cache)
    else:
        parts = collision_parts(np.asarray(geometry.vertices)*mesh.scale,geometry.faces,
                                tolerance=.00025 if 'finger' in g.parent.name else .001,
                                surface_thickness=.0002,cache=args.cache)
    for i,(vertices,faces) in enumerate(parts):
        name = f'{g.parent.name}_collision_{i}'
        s.add_mesh(name=name,uservert=vertices.ravel(),userface=faces.ravel(),
                   inertia=mj.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        g.parent.add_geom(name=name,type=mj.mjtGeom.mjGEOM_MESH,meshname=name,
                          pos=g.pos,quat=g.quat,group=3,friction=g.friction,condim=4,density=0)
    g.contype = g.conaffinity = 0
    rows.append({'body':g.parent.name,'convex_parts':len(parts),'seconds':time.monotonic()-t})
    print(rows[-1],flush=True)

limits = {}
for joint in root.findall('joint'):
    lim = joint.find('limit')
    if lim is None or joint.get('type') == 'fixed':
        continue
    name = joint.get('name')
    effort = float(lim.get('effort'))
    limits[name] = {'effort':effort,'velocity':float(lim.get('velocity'))}
    if joint.find('mimic') is not None:
        continue
    operating_effort = min(effort,5.) if 'finger_axis' in name else effort
    a = s.add_actuator(name=name,target=name,trntype=mj.mjtTrn.mjTRN_JOINT,
                       forcelimited=True,forcerange=[-operating_effort,operating_effort])
    if '_wheel_joint' in name or '_steering_joint' in name:
        a.gainprm = [1]+[0]*9
    else:
        kp,kv = ((80000,3000) if 'torso_lift' in name else
                 (2000,100) if 'waist_pitch' in name else
                 (3000,40) if 'finger_axis' in name else (effort*25,max(2,effort)))
        a.gainprm = [kp]+[0]*9
        a.biastype = mj.mjtBias.mjBIAS_AFFINE
        a.biasprm = [0,-kp,-kv]+[0]*7
        a.ctrllimited = True
        a.ctrlrange = [float(lim.get('lower')),float(lim.get('upper'))]

for side in ('left','right'):
    s.body(f'{side}_gripper_link').add_site(name=f'{side}_tcp',pos=[0,0,-.115],quat=[0,2**-.5,2**-.5,0],group=4)
    s.body(f'{side}_gripper_camera_link').add_camera(name=f'{side}_wrist',quat=[0,1,0,0],fovy=85)
camera_quat=np.zeros(4)
mj.mju_mat2Quat(camera_quat,np.array([[0,-1,0],[0,0,1],[-1,0,0]],dtype=float).ravel())
for name,y in (('head',.03),('head_right',-.03)):
    s.body('stereo_camera_link').add_camera(name=name,pos=[.018,y,0],quat=camera_quat,fovy=87)
for g in s.geoms:
    g.solref=[.004,1];g.solimp=[.95,.99,.001,.5,2]
for eq in s.equalities:
    eq.solref=[.004,1]; eq.solimp=[.99,.999,.001,.5,2]
m = s.compile()
assert abs(float(m.body_mass.sum())-42.9194829)<1e-6
q = m.qpos0.copy()
pose = {'torso_lift_joint':0,'waist_pitch_joint':0,'head_pitch_joint':.314}
pose.update({f'{side}_arm_joint_{i}':0 for side in ('left','right') for i in range(1,8)})
pose.update({j:.04 for j in limits if 'finger_axis' in j})
for name,value in pose.items():
    q[m.joint(name).qposadr[0]] = value
ctrl = np.zeros(m.nu)
for i in range(m.nu):
    name = m.actuator(i).name
    ctrl[i] = pose.get(name,0)
s.add_key(name='home',qpos=q,ctrl=ctrl)
m = s.compile()
(out/'physical.xml').write_text(s.to_xml())
shutil.copyfile(Path(__file__).resolve().parents[1]/'examples/household/embodiments/malo/robot.yaml', out/'robot.yaml')
mj.mj_saveModel(m,str(out/'physical.mjb'))
json.dump({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'mass_kg':float(m.body_mass.sum()),
           'collision_parts':rows,'urdf_limits':limits,'home':pose,
           'collision_algorithm':'convex-parts-v3, cavity-preserving; physical acceptance required',
           'source':str(source), 'mujoco_version':mj.__version__,
           'source_mesh_sha256':{str(Path(mesh.get('filename'))):hashlib.sha256(Path(mesh.get('filename')).read_bytes()).hexdigest() for mesh in root.iter('mesh')},
           'estimated':['servo gains','contact friction','0.1 mm base / 0.2 mm wrist open-surface thickness'],
           'sensor_calibration':'CAD frames; rectified pinhole head FOV 87deg, wrist FOV 85deg and head eye offsets are engineering assumptions, not calibrated hardware intrinsics',
           'software_grip_force_cap_N':5.,'contact_solref':[.004,1],'contact_solimp':[.95,.99,.001,.5,2],
           'cad_redistribution':'not cleared; local references only'},(out/'audit.json').open('w'),indent=2)
d = mj.MjData(m)
mj.mj_resetDataKeyframe(m,d,0)
mj.mj_forward(m,d)
assert m.njnt == 31 and m.nq == 37 and m.nu == 28
assert all(m.actuator_forcelimited)
print('BUILT',m.nbody,m.njnt,m.ngeom,'mass',m.body_mass.sum(),'contacts',d.ncon,flush=True)
for c in sorted(d.contact,key=lambda x:x.dist)[:8]:
    print('CONTACT',c.dist,m.body(m.geom_bodyid[c.geom1]).name,m.body(m.geom_bodyid[c.geom2]).name,flush=True)
