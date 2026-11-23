"""Read recorded messages; compare applied velocity with the installed controller."""
from collections import deque, Counter
from pathlib import Path
import json, math, sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

folder=Path(sys.argv[1])
reader=rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=str(folder/'nav-bag'),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
types={t.name:t.type for t in reader.get_all_topics_and_types()}
needed={'/navigation_cmd_vel','/vector/navigation/applied_cmd','/vector/navigation/status','/odom_base_link'}
classes={name:get_message(types[name]) for name in needed}
raw=deque(maxlen=200);matches=Counter();unmatched=[];counts=Counter();positions=[];odom=[];stops=[]
planner='far';state='idle'
def stamp(h):return h.stamp.sec+h.stamp.nanosec/1e9
def vector(m):return (m.twist.linear.x,m.twist.linear.y,m.twist.angular.z)
while reader.has_next():
 topic,data,wall=reader.read_next()
 if topic not in needed:continue
 m=deserialize_message(data,classes[topic]);counts[topic]+=1
 if topic=='/vector/navigation/status':
  row=json.loads(m.data);planner=row['planner']
  if row['state']=='aborted' and state!='aborted':stops.append(row['sim_time'])
  state=row['state']
 elif topic=='/navigation_cmd_vel':
  assert m.header.frame_id=='base_link'
  raw.append((wall,tuple(max(-lim,min(lim,v)) for v,lim in zip(vector(m),(.15,.15,.2)))))
 elif topic=='/vector/navigation/applied_cmd':
  values=vector(m)
  assert m.header.frame_id=='base_link' and all(math.isfinite(v) for v in values)
  assert all(abs(v)<=lim+1e-9 for v,lim in zip(values,(.15,.15,.2)))
  if max(map(abs,values))>1e-5:
   match=any(0<=wall-t<=2e9 and max(abs(a-b) for a,b in zip(values,other))<1e-9 for t,other in raw)
   matches[planner]+=int(match)
   if not match:unmatched.append({'stamp':stamp(m.header),'values':values})
 else:
  p=m.pose.pose.position;positions.append((p.x,p.y));v=m.twist.twist.linear
  odom.append((stamp(m.header),math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)))
del reader
delays=[]
for stop in stops:
 found=next((t-stop for t,speed in odom if stop<=t<=stop+3 and speed<.03),None)
 delays.append({'aborted_state_sim_time':stop,'time_until_speed_below_0_03_m_s':found})
assert positions and sum(matches.values())>0
report={'topic_counts':dict(counts),'nonzero_applied_matches':dict(matches),'unmatched_nonzero_count':len(unmatched),
        'unmatched_examples':unmatched[:5],'odometry_travel_m':sum(math.dist(a,b) for a,b in zip(positions,positions[1:])),
        'stop_delays_sim_seconds':delays,'all_nonzero_commands_match_recorded_pathFollower':not unmatched}
(folder/'command-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
assert not unmatched
