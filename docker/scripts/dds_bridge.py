from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader, Qos, Listener, Subscriber
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.domain import Domain, DomainParticipant
from unitree_sdk2py.idl.unitree_go.msg.dds_._MotorStates_ import *
from unitree_sdk2py.idl.unitree_hg.msg.dds_._LowState_ import *
from unitree_sdk2py.idl.unitree_go.msg.dds_._MotorCmds_ import *
from unitree_sdk2py.idl.unitree_hg.msg.dds_._LowCmd_ import *
from unitree_sdk2py.idl.std_msgs.msg.dds_._String_ import String_
import os
# DOMAIN_ID = int(os.getenv("ROS_DOMAIN_ID"))
# assert DOMAIN_ID > 0, "Please set ROS_DOMAIN_ID environment variable to a positive integer for DDS channel factory initialization, domain id 0 reserved for real robot"
DOMAIN_ID = 0


dp_isaac = DomainParticipant(DOMAIN_ID+1)
dp_cyclone = DomainParticipant(DOMAIN_ID)
qos = Qos()
listener = Listener()
sub_isaac = Subscriber(dp_isaac, qos, listener)
sub_cyclone = Subscriber(dp_cyclone, qos, listener)
pub_isaac = Publisher(dp_isaac, qos, listener)
pub_cyclone = Publisher(dp_cyclone, qos, listener)
bridges = [
    ("lowstate", "rt/lowstate", LowState_, "isaac", "cyclone"),
    ("inspire_state", "rt/inspire/state", MotorStates_, "isaac", "cyclone"),
    ("sim_state", "rt/sim_state", String_, "isaac", "cyclone"),
    ("rewards_state", "rt/rewards_state", String_, "isaac", "cyclone"),
    ("inspire_cmd", "rt/inspire/cmd", MotorCmds_, "cyclone", "isaac"),
    ("low_cmd", "rt/lowcmd", LowCmd_, "cyclone", "isaac"),
    ("reset_pose_cmd", "rt/reset_pose/cmd", String_, "cyclone", "isaac")]
participants = {"isaac": (dp_isaac, sub_isaac, pub_isaac), "cyclone": (dp_cyclone, sub_cyclone, pub_cyclone)}
readers = {}
writers = {}
for name, topic_str, msg_type, read_domain, write_domain in bridges:
    read_dp,  read_sub,  _  = participants[read_domain]
    write_dp, _, write_pub = participants[write_domain]
    read_topic  = Topic(read_dp,  topic_str, msg_type, qos, listener)
    write_topic = Topic(write_dp, topic_str, msg_type, qos, listener)
    readers[name] = DataReader(read_sub,  read_topic,  qos, listener)
    writers[name] = DataWriter(write_pub, write_topic, qos, listener)


    print(f"\n\n\nBridge set up for topic '{topic_str}' with message type '{msg_type.__name__}' from '{read_domain}' to '{write_domain}' \n\n\n")

import asyncio 
from datetime import datetime
async def main():
    while True:
        ts = datetime.now()
        for name in readers:
            try:
                writers[name].write(readers[name].read()[0])
            except Exception as e:
                pass
        await asyncio.sleep(1)
asyncio.run(main())
