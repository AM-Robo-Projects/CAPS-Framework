from ultralytics import YOLO
import cv2
import numpy as np
#import rembg
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError 


#TODO

MODEL_PATH = '/home/leitrechner/CAPS-Framework-/V1_640_epochs50_yolov8l_BEST/best.pt'

class TruckObjectDetection:
    """!
    @brief does the object detection and analysis of the image and publishes the object confidence scores to decision making and grasping nodes 
    """
    def __init__(self,MODEL_PATH,rgbImageTopic):
        """!
        @brief contructor

        Parameters : 
            @param self => object of the class
            @param MODEL_PATH => Trained model path to detect truck parts 

        """
        rospy.init_node ("TruckObjectDetection")
        
        self._imageSubscriber = rospy.Subscriber(rgbImageTopic,Image,self.getImage,queue_size=2)
        self._objectDetectionImagePublsiher = rospy.Publisher("/Object_detection_images",Image,queue_size=2)
       
        
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)

        self.bridge = CvBridge()
        self._truckObjectDetection = YOLO(MODEL_PATH)
        self.DETECTION_THRESHOLD = 0.6
        self.DESICION_THRESHOLD = 0.6

        self._objDetImg = None
        self._resimgDetection = np.zeros((1440, 1080),dtype=np.int16)
        self._rgbImg = None
        print("preping models")
        self.prepModels()
        
        
    def getImage(self, img): 
        """
        @brief gets the image from the topic

        """
        
        try:
            self._rgbImg = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8") 
        except CvBridgeError as e: 
            
            self._rgbImg = None
            
    def prepModels(self):
        """!
        @brief prepares model for later use, speeds up inference later

        Parameters : 
            @param self => object of the class

        """
        img = np.zeros((1440, 1080, 3), dtype=np.int16)
        self._truckObjectDetection(img, verbose=False)[0].cpu()
   

    def runLocally(self, inDir, outDir):
        """!
        @brief run based on local pictures for testing

        Parameters : 
            @param self => object of the class
            @param inDir => directory path of the pictures
            @param outDir => directory path where the results should be stored

        """
        pics = os.listdir(inDir)
        pics.sort()
        os.makedirs(outDir, exist_ok= True)
        #os.makedirs(os.path.join(outDir, "cnt"), exist_ok= True)
        for entry in pics:
            img = cv2.imread(os.path.join(inDir,entry))

            try:
                self.analyseOneImg(img)
                #cv2.imwrite(os.path.join(outDir, entry), self._resimgBending)
            except RuntimeError as e : 
                print(f"couldn't find objects {e}")

        return
    
    
    def runRos(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self._rgbImg is not None:
                self.analyseOneImg(self._rgbImg)
               # self._objectDetectionImagePublsiher (self._objDetImg)
            else:
                rospy.logwarn("No image received yet. Waiting...")
            rate.sleep()

    
    def analyseOneImg(self, img):
        """!
        @brief analysis of one image

        Parameters : 
            @param self => object of the class
            @param img => np array of the RGB image
        """
        if img is None:
            rospy.logwarn("Input image is None. Skipping analysis.")
            return

        # Run YOLO object detection
        try:
            results = self._truckObjectDetection(img, verbose=False)[0]
        except Exception as e:
            rospy.logerr(f"Error during object detection: {e}")
            return

        # Process results
        self._resimgDetection = results.plot()
        self._boxes = results.boxes.xyxy.cpu().numpy().reshape((-1, 2, 2))  
        self._predictions = results.boxes.conf.cpu().numpy()            
        self._classes = results.boxes.cls.cpu().numpy()                    
        self.filterDetection(self.DETECTION_THRESHOLD)
        self._objDetImg = self.vizDetections(img)
        
        #TODO define the published image should contain the undesired object or the high confidence object.
        try:
            ros_image = self.bridge.cv2_to_imgmsg(self._objDetImg, encoding="bgr8")
            self._objectDetectionImagePublsiher.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting OpenCV image to ROS Image: {e}")
        # Display the image
        cv2.imshow("test", self._objDetImg)
        cv2.waitKey(1)

   
    #TODO Add ImgPublisher function to remove bbox and keep white filled box and publish to grasping node     
    
    #Threshold 
    def filterDetection(self, minPred):
        """!
        @brief filteres the detection based on a minimal theshold precision score

        Parameters : 
            @param self => object of the class
            @param minPred => minimal theshold precision score

        """
        decisionArr = np.array(self._predictions > minPred)
        self._boxes = self._boxes[decisionArr]
        self._predictions = self._predictions[decisionArr]
        self._classes = self._classes[decisionArr]
    
    #Gets the center of a bounding box
    def getbbCenter(bb):
        """!
        @brief returns the center of a given bounding box in the format [[upper left x, upper left y], [lower right x, lower right y]]

        Parameters : 
            @param bb => array of the bounding box

        """
        x_center = (bb[0,0] + bb[1,0] )/ 2
        y_center = (bb[0,1] + bb[1,1] )/ 2
        return np.array([x_center, y_center])
 

    def vizDetections(self, img):
        """!
        @brief Visualizes the detections stored in the object's attributes.

        Parameters:
            @param self => Object of the class
            @param img => Image to draw onto
        """
        img = img.copy()
        class_names = ["Truck Cabin", "Truck Loader"]  # Map class IDs to names
        for index, box in enumerate(self._boxes):
            class_id = int(self._classes[index])  # Convert class to int
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"  # Handle invalid IDs

            # Determine action based on confidence score
            confidence = self._predictions[index]
            
            if confidence < self.DESICION_THRESHOLD :
                # Fill the bounding box with white to remove the object
                start_point = box[0].astype(int)
                end_point = box[1].astype(int)
                img[start_point[1]:end_point[1], start_point[0]:end_point[0]] = (255, 255, 255)  # White fill
                        
            else:
                # Draw the bounding box and class name for high-confidence detections
                color = (150, 150, 255) if class_id == 0 else (150, 255, 150)  # Color for bounding box
                img = TruckObjectDetection.drawBBonImg(img, box, color, f"{class_name} ({confidence:.2f})")

        return img




    def drawBBonImg(img, BB, color=(36, 255, 12), class_name="Unknown"):
        """!
        @brief Draws a bounding box and class name on the given image.

        Parameters:
            @param img => Image to draw onto
            @param BB => Bounding box in the format [[upper left x, upper left y], [lower right x, lower right y]]
            @param color => Color of the bounding box (default is green)
            @param class_name => Name of the class to display (default is "Unknown")
        """
        start_point = BB[0].astype(int)
        end_point = BB[1].astype(int)
        thickness = 3
        image = cv2.rectangle(img, start_point, end_point, color, thickness)

        # Add class name and confidence score above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
        text_origin = (start_point[0], start_point[1] - 10)

        # Draw a background rectangle for the text for better visibility
        cv2.rectangle(
            img,
            (text_origin[0], text_origin[1] - text_size[1] - 5),
            (text_origin[0] + text_size[0], text_origin[1] + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(img, class_name, text_origin, font, font_scale, (255, 255, 255), font_thickness)
        return image


    
if __name__ == "__main__":
    rgb_topic = "/camera/color/image_raw"
    cable = TruckObjectDetection(MODEL_PATH,rgb_topic)
    cable.runRos()