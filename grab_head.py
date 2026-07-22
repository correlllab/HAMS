import rclpy, numpy as np, cv2, time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
class G(Node):
    def __init__(self):
        super().__init__('grab_head'); self.done=False
        self.create_subscription(CompressedImage,'/realsense/head/color/image_raw/compressed',self.cb,qos_profile_sensor_data)
    def cb(self,m):
        arr=np.frombuffer(m.data,np.uint8); img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        cv2.imwrite('/tmp/head_view.jpg',img); print('saved',img.shape); self.done=True
rclpy.init(); n=G(); t=time.time()
while not n.done and time.time()-t<8: rclpy.spin_once(n,timeout_sec=0.2)
rclpy.shutdown()
