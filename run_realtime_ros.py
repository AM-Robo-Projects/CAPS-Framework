import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results
from utils.dataset_processing.grasp import detect_grasps
from hardware.device import get_device

logging.basicConfig(level=logging.INFO)

bridge = CvBridge()
rgb_image = None
depth_image = None

def rgb_callback(msg):
    global rgb_image
    try:
        rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting RGB image: {e}")

def depth_callback(msg):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting depth image: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default='',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    return parser.parse_args()

def run():
    global rgb_image, depth_image

    args = parse_args()
    rospy.init_node('gr_net_node')

    # Replace topics with Kinect Azure's topics
    rospy.Subscriber('/kinect1/color/image_raw', Image, rgb_callback)
    rospy.Subscriber('/kinect1/depth/image_raw', Image, depth_callback)
    grasp_publisher = rospy.Publisher("/result", Float64MultiArray, queue_size=10)

    # Prepare data processor
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)
    cam_depth_scale = 1

    camera2robot = np.array(
        [[-0.36037981,  0.77589333, -0.51779912, 0.73501246],
         [0.93190563,  0.27509038, -0.23638353, 0.21445706],
         [-0.04096684, -0.56772777, -0.82219639, 0.735518],
         [0., 0., 0., 1.]]
    )

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    device = get_device(args.force_cpu)
    logging.info('Model loaded.')

    try:
        fig = plt.figure(figsize=(10, 10))
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if rgb_image is None or depth_image is None:
                rospy.loginfo("Waiting for images...")
                rate.sleep()
                continue

            # Process images
            x, depth_img, rgb_img = cam_data.get_data(rgb=rgb_image, depth=depth_image)
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                grasps = detect_grasps(q_img, ang_img, width_img)

                # Grasp position calculation
                pos_z = depth_image[grasps[0].center[0] + cam_data.top_left[0], grasps[0].center[1] + cam_data.top_left[1]] * cam_depth_scale
                pos_x = np.multiply(grasps[0].center[1] + cam_data.top_left[1] - cam_data.ppx,
                                    pos_z / cam_data.fx)
                pos_y = np.multiply(grasps[0].center[0] + cam_data.top_left[0] - cam_data.ppy,
                                    pos_z / cam_data.fy)
                target = np.asarray([pos_x, pos_y, pos_z])
                target.shape = (3, 1)

                target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
                target_position = target_position[0:3, 0]
                target_angle = grasps[0].angle + (np.pi / 6)
                width = grasps[0].width

                grasp_pose = np.append(target_position, target_angle)
                grasp = Float64MultiArray()
                grasp.data = grasp_pose
                grasp_publisher.publish(grasp)

                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb_image, False),
                             depth_img=np.squeeze(cam_data.get_depth(depth_image)),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)
                rate.sleep()
    except rospy.ROSInterruptException:
        logging.info("Shutting down ROS node.")
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}")

if __name__ == '__main__':
    run()

