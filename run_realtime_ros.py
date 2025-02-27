import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2

from inference.post_process import post_process_output
from utils.data.camera_data_ros import CameraData
from utils.visualisation.plot import save_results, plot_results, plot_grasp
from utils.dataset_processing.grasp import detect_grasps
from hardware.device import get_device



class GraspDetector :

    def __init__(self,Rgb_topic,Depth_topic) :
        

        
        rospy.init_node('gr_net_node',anonymous=False)
        self._rgb_sub = rospy.Subscriber(Rgb_topic, Image, self.rgb_callback)
        self._depth_sub = rospy.Subscriber(Depth_topic, Image, self.depth_callback)
        #self._depth_sub = rospy.Subscriber(Depth_topic, PointCloud2, self.depth_callback)

        self.grasp_publisher = rospy.Publisher("/grasp_pose", Float64MultiArray, queue_size=10)
        

        

        self._bridge = CvBridge()
        self._rgb_image = None
        self._depth_image = None

    def rgb_callback(self,msg):
       
        try:
            self._rgb_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            print(f"RGB : {self._rgb_image.shape}")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting RGB image: {e}")

    def depth_callback(self,msg):
        
        try:
            self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")


        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image: {e}")

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Evaluate network')
        parser.add_argument('--network', type=str, default='/home/abdelrahman/CAPS-Framework/trained-models/epoch_08_iou_1.00',
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

    def run(self):
        
       
        args = self.parse_args()
        logging.basicConfig(level=logging.INFO)

        print("getting_cameraData")
        # Prepare data processor
        try : 
            cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)
            print ("data retrived...")
        except RuntimeError as e : 
            print(e)


        cam_depth_scale = 0.001

        camera2robot = np.array([
                                            [0.6927,  0.0172, -0.7210,  0.26677],
                                            [-0.0402, 0.9991, -0.0148,  0.45252],
                                            [0.7201,  0.0393,  0.6928,  -0.32341],
                                            [0.0000,  0.0000,  0.0000,  1.0000]
                                        ])

        # Load Network
        #logging.info('Loading model...')
        print ("loading model")
        #net = torch.load(args.network)
        net = torch.load(args.network, map_location=torch.device('cpu'))
        device = get_device(args.force_cpu)
        print ("model loaded")
        #logging.info('Model loaded.')

        try:
            fig = plt.figure(figsize=(10, 10))
            print ("plotting")
            rate = rospy.Rate(10)

            while not rospy.is_shutdown():
                if self._rgb_image is None or self._depth_image is None:
                    rospy.loginfo("Waiting for images...")
                    rate.sleep()
                    continue

                # Process images
                x, depth_img, rgb_img = cam_data.get_data(rgb=self._rgb_image, depth=self._depth_image)
                print("images_processed")
                
                with torch.no_grad():
                    print("entered loop")
                    
                    xc = x.to(device)
                    xc = xc.unsqueeze(0)
                    print(xc.shape)  # Should print: torch.Size([1, channels, height, width])

                   # print("device initiated")
                    
                    try: 

                        pred = net.predict(xc)
                    except RuntimeError as e :
                        print(e)

                    print("prediction done")
                    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                    grasps = detect_grasps(q_img, ang_img, width_img)
            
                    try : 
                    # Grasp position calculation
                    

                        pos_z = self._depth_image[grasps[0].center[0] + cam_data.top_left[0], grasps[0].center[1] + cam_data.top_left[1]] * cam_depth_scale 
                        
                        pos_x = np.multiply(grasps[0].center[1] + cam_data.top_left[1] - cam_data.ppx,
                                    pos_z / cam_data.fx)
                    
                        pos_y = np.multiply(grasps[0].center[0] +cam_data.top_left[0] - cam_data.ppy,
                                    pos_z / cam_data.fy)
                        
                    except RuntimeError as e :
                        print (f"ERROR : {e}")


                   
                    target = np.asarray([pos_x, pos_y, pos_z])
                    
                    target.shape = (3, 1)
                    print('target: ', target)

                    target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
                    target_position = target_position[0:3, 0]

                    #angle = np.asarray([0, 0, grasps[0].angle])
                    #angle.shape = (3, 1)
                    #target_angle = np.dot(camera2robot[0:3, 0:3], angle) 
                    target_angle = grasps[0].angle + (np.pi/6)

                    width = grasps[0].width
                    print(width)

                    grasp_pose = grasp_pose = np.append(target_position, [target_angle, width])
                    # x,y,z, rotation around z and width of gripper
                    print (grasp_pose)
                    grasp = Float64MultiArray()
                    grasp.data = grasp_pose
                    
                    self.grasp_publisher.publish(grasp)

                     #plot_grasp(fig=fig,
                #            rgb_img=cam_data.get_rgb(rgb, False),                 
                #            grasps=grasps,
                #            save=False)

                    plot_results(fig=fig,
                                rgb_img =cam_data.get_rgb(self._rgb_image, False),
                                depth_img=np.squeeze(cam_data.get_depth(self._depth_image)),
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                no_grasps=args.n_grasps,
                                grasp_width_img=width_img)
                    rate.sleep()
            rospy.spin()

        except rospy.ROSInterruptException:
            logging.info("Shutting down ROS node.")
        except RuntimeError as e:
            logging.error(f"Runtime error: {e}")

if __name__ == '__main__':
    
    rgb_topic = '/camera/color/image_raw'
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    Grasp = GraspDetector(rgb_topic,depth_topic)
    Grasp.run()
    

