from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader, Qos, Listener, Subscriber
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.domain import Domain, DomainParticipant
from unitree_sdk2py.idl.unitree_go.msg.dds_._MotorStates_ import *
from unitree_sdk2py.idl.unitree_hg.msg.dds_._LowState_ import *
from unitree_sdk2py.idl.unitree_go.msg.dds_._MotorCmds_ import *
from unitree_sdk2py.idl.unitree_hg.msg.dds_._LowCmd_ import *
from unitree_sdk2py.idl.std_msgs.msg.dds_._String_ import String_

dp_1 = DomainParticipant(1)
dp_2 = DomainParticipant(0)
qos = Qos()
listener = Listener()
sub_d1 = Subscriber(dp_1, qos, listener)
sub_d2 = Subscriber(dp_2, qos, listener)
pub_d1 = Publisher(dp_1, qos, listener)
pub_d2 = Publisher(dp_2, qos, listener)
bridges = [
    ("lowstate", "rt/lowstate", LowState_, 1, 2),
    ("inspire_state", "rt/inspire/state", MotorStates_, 1, 2),
    ("sim_state", "rt/sim_state", String_, 1, 2),
    ("rewards_state", "rt/rewards_state", String_, 1, 2),
    ("inspire_cmd", "rt/inspire/cmd", MotorCmds_, 2, 1),
    ("low_cmd", "rt/lowcmd", LowCmd_, 2, 1),
    ("reset_pose_cmd", "rt/reset_pose/cmd", String_, 2, 1)]
participants = {1: (dp_1, sub_d1, pub_d1), 2: (dp_2, sub_d2, pub_d2)}
readers = {}
writers = {}
for name, topic_str, msg_type, read_domain, write_domain in bridges:
    read_dp,  read_sub,  _  = participants[read_domain]
    write_dp, _, write_pub = participants[write_domain]
    read_topic  = Topic(read_dp,  topic_str, msg_type, qos, listener)
    write_topic = Topic(write_dp, topic_str, msg_type, qos, listener)
    readers[name] = DataReader(read_sub,  read_topic,  qos, listener)
    writers[name] = DataWriter(write_pub, write_topic, qos, listener)

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
