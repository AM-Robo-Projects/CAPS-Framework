import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import rospy

from sensor_msgs.msg import Image 
from std_msgs.msg import Float64MultiArray
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results, plot_grasp
from utils.dataset_processing.grasp import detect_grasps

logging.basicConfig(level=logging.INFO)


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

        args = parser.parse_args()
        return args


        

def run():
        

        args = parse_args()
        
        # Connect to Camera
        logging.info('Connecting to camera...')
        cam = RealSenseCamera(device_id='247122070300')
        cam.connect()
        cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)
        cam_depth_scale = 1 
        
        camera2robot = np.array(
                    [[-0.36037981,  0.77589333, -0.51779912, 0.73501246],
                    [ 0.93190563,  0.27509038, -0.23638353,  0.21445706],
                    [-0.04096684, -0.56772777, -0.82219639,  0.735518  ],
                    [ 0.,          0.,          0.,          1.        ]])


    



        # Load Network
        logging.info('Loading model...')
        net = torch.load(args.network)
        logging.info('Done')

        # Get the compute device
        device = get_device(args.force_cpu)

        try:
            
            fig = plt.figure(figsize=(10, 10))
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                image_bundle = cam.get_image_bundle()
                rgb = image_bundle['rgb']
                depth = image_bundle['aligned_depth']
                x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
                with torch.no_grad():
                    xc = x.to(device)
                    pred = net.predict(xc)

                    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                    grasps = detect_grasps(q_img, ang_img, width_img)

                    pos_z = depth[grasps[0].center[0] + cam_data.top_left[0], grasps[0].center[1] + cam_data.top_left[1]] * cam_depth_scale # -0.04
                    pos_x = np.multiply(grasps[0].center[1] + cam_data.top_left[1] - cam.intrinsics.ppx,
                                pos_z / cam.intrinsics.fx)
                    pos_y = np.multiply(grasps[0].center[0] +cam_data.top_left[0] - cam.intrinsics.ppy,
                                pos_z / cam.intrinsics.fy)
                    target = np.asarray([pos_x, pos_y, pos_z])
                    target.shape = (3, 1)
                    #print('target: ', target)

                    target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
                    target_position = target_position[0:3, 0]

                    #angle = np.asarray([0, 0, grasps[0].angle])
                    #angle.shape = (3, 1)
                    #target_angle = np.dot(camera2robot[0:3, 0:3], angle) 
                    target_angle = grasps[0].angle + (np.pi/6)

                    width = grasps[0].width
                    print(width)


                    grasp_pose = np.append(target_position, target_angle)
                    
                    grasp = Float64MultiArray()
                    grasp.data = grasp_pose
                    print('grasp_pose: ', grasp.data)
                    grasp_publisher.publish(grasp)
                    
                    
                    #plot_grasp(fig=fig,
                    #            rgb_img=cam_data.get_rgb(rgb, False),                 
                    #            grasps=grasps,
                    #            save=False)
                    
                    plot_results(fig=fig,
                                rgb_img=cam_data.get_rgb(rgb, False),
                                depth_img=np.squeeze(cam_data.get_depth(depth)),
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                no_grasps=args.n_grasps,
                                grasp_width_img=width_img)
                    
                    rate.sleep()
        
        except RuntimeError as e :
            print(e)
        #finally:
        #    save_results(
        #        rgb_img=cam_data.get_rgb(rgb, False),
        #        depth_img=np.squeeze(cam_data.get_depth(depth)),
        #        grasp_q_img=q_img,
        #        grasp_angle_img=ang_img,
        #        no_grasps=args.n_grasps,
        #        grasp_width_img=width_img 
        #    )
                
if __name__ == '__main__':

    try:
        rospy.init_node('gr_net_node')
        grasp_publisher = rospy.Publisher("/result",Float64MultiArray,queue_size=10)
        run()
        rospy.spin()
    except RuntimeError as e :
        print (e)    
