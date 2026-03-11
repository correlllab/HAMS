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

lowstate_topic_d1 = Topic(dp_1, "rt/lowstate", LowState_, qos, listener=listener)
lowstate_topic_d2 = Topic(dp_2, "rt/lowstate", LowState_, qos, listener)
lowcmd_topic_d1 = Topic(dp_1, "rt/lowcmd", LowCmd_, qos, listener)
lowcmd_topic_d2 = Topic(dp_2, "rt/lowcmd", LowCmd_, qos, listener)
sim_state_topic_d1 = Topic(dp_1, "rt/sim_state", String_, qos=qos, listener=listener)
sim_state_topic_d2 = Topic(dp_2, "rt/sim_state", String_, qos, listener)
inspire_state_topic_d1 = Topic(dp_1, "rt/inspire/state", MotorStates_, qos=qos, listener=listener)
inspire_state_topic_d2 = Topic(dp_2, "rt/inspire/state", MotorStates_, qos, listener)
rewards_state_topic_d1 = Topic(dp_1, "rt/rewards_state", String_, qos, listener)
rewards_state_topic_d2 = Topic(dp_2, "rt/rewards_state", String_, qos, listener)
inspire_cmd_topic_d1 = Topic(dp_1, "rt/inspire/cmd", MotorCmds_, qos, listener)
inspire_cmd_topic_d2 = Topic(dp_2, "rt/inspire/cmd", MotorCmds_, qos, listener)
reset_pose_cmd_topic_d1 = Topic(dp_1, "rt/reset_pose/cmd", String_, qos, listener)
reset_pose_cmd_topic_d2 = Topic(dp_2, "rt/reset_pose/cmd", String_, qos, listener)

lowstate_dr_d1 = DataReader(sub_d1, lowstate_topic_d1, qos, listener)
lowstate_dw_d2 = DataWriter(pub_d2, lowstate_topic_d2, qos, listener)
inspire_state_dr_d1 = DataReader(sub_d1, inspire_state_topic_d1, qos, listener)
inspire_state_dw_d2 = DataWriter(pub_d2, inspire_state_topic_d2, qos, listener)
sim_state_dr_d1 = DataReader(sub_d1, sim_state_topic_d1, qos, listener)
sim_state_dw_d2 = DataWriter(pub_d2, sim_state_topic_d2, qos, listener)
rewards_state_dr_d1 = DataReader(sub_d1, rewards_state_topic_d1, qos, listener)
rewards_state_dw_d2 = DataWriter(pub_d2, rewards_state_topic_d2, qos, listener)
inspire_cmd_dr_d2 = DataReader(sub_d2, inspire_cmd_topic_d2, qos, listener)
inspire_cmd_dw_d1 = DataWriter(pub_d1, inspire_cmd_topic_d1, qos, listener)
low_cmd_dr_d2 = DataReader(sub_d2, lowcmd_topic_d2, qos, listener)
low_cmd_dw_d1 = DataWriter(pub_d1, lowcmd_topic_d1, qos, listener)
low_cmd_dw_d2 = DataWriter(pub_d2, lowcmd_topic_d2, qos, listener)
reset_pose_cmd_dw_d1 = DataWriter(pub_d1, reset_pose_cmd_topic_d1, qos ,listener)
reset_pose_cmd_dr_d2 = DataReader(sub_d2, reset_pose_cmd_topic_d2, qos, listener)

import asyncio
async def main():
    from datetime import datetime
    while True:
        current_datetime = datetime.now()
        try:
            msg = lowstate_dw_d2.write(lowstate_dr_d1.read()[0])
        except Exception as e:
            print(f"{current_datetime}: how to gNo data from rt/lowstate: {e}")
        try:
            msg = low_cmd_dw_d1.write(low_cmd_dr_d2.read()[0])
        except Exception as e:
            print(f"{current_datetime}: No data from rt/lowcmd: {e}")
        try:
            msg = inspire_cmd_dw_d1.write(inspire_cmd_dr_d2.read()[0])
        except Exception as e:
            print(f"{current_datetime}: No data from rt/inspire/cmd: {e}")
        try:
            msg = rewards_state_dw_d2.write(rewards_state_dr_d1.read()[0])
        except Exception as e:
            print(f"{current_datetime}: No data from rt/rewards_state: {e}")
        try:
            msg = sim_state_dw_d2.write(sim_state_dr_d1.read()[0])
        except Exception as e:
            print(f"{current_datetime}: No data from rt/sim_state: {e}")
        try:
            msg = reset_pose_cmd_dw_d1.write(reset_pose_cmd_dr_d2.read()[0])
        except Exception as e:
            print(f"{current_datetime}: No data from rt/reset_pose/cmd: {e}")
        try:
            msg = inspire_state_dw_d2.write(inspire_state_dr_d1.read()[0])
        except Exception as e:
            print(f"{current_datetime}: No data from rt/inspire/state: {e}")
        await asyncio.sleep(1)

asyncio.run(main())

