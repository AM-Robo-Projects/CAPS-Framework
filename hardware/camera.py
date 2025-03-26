# filepath: /home/abdelrahman/CAPS-Framework/hardware/camera.py
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

logger = logging.getLogger(__name__)

class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None
        self.bridge = CvBridge()
        #rospy.init_node('camera', anonymous=True)

        self.rgb_pub = rospy.Publisher('/Image/color/image_raw', Image, queue_size=10)
        

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)

        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        
        color_image = np.asanyarray(color_frame.get_data())  
        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def publish_images(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        try:
            print ("going in the color image ")
            ros_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
            self.rgb_pub.publish(ros_rgb)
           
        except CvBridgeError as e:
            rospy.logerr(f"Error converting images: {e}")

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()

if __name__ == '__main__':
    cam = RealSenseCamera(device_id=247122070300)
    cam.connect()
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        cam.publish_images()
        rate.sleep()