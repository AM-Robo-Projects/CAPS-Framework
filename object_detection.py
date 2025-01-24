from ultralytics import YOLO
import cv2
import numpy as np
import rembg
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 


#TODO

MODEL_PATH = '/home/abdelrahman/robotic-grasping-CNN_1/V1_640_epochs50_yolov8l_BEST/best.pt'

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
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)

        self._cv_bridge = CvBridge()
        self._truckObjectDetection = YOLO(MODEL_PATH)
        self.DETECTION_THRESHOLD = 0.6

        self._objDetImg = None
        self._resimgDetection = np.zeros((1440, 1080),dtype=np.int16)

        print("preping models")
        self.prepModels()
        
        
        
    def getImage (self,img): 
        try :
           self._rgbImg = self._cv_bridge.imgmsg_to_cv2(img,desired_encoding="passthrough") 
        except CvBridgeError as e : 
            rospy.loginfo(e)
            
            
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
            #outImg = self.bgremove2(img)
            #outImg = rembg.remove(img, post_process_mask=True)
            #try:
            # results = self._model(img, verbose=False)[0].cpu()
            # #self._resimg = results.plot()
            # self._boxes = results.boxes.xyxy.numpy().reshape((-1,2,2))
            # self._predictions = results.boxes.conf.numpy()
            # self._classes = results.boxes.cls.numpy()
            # self.filterDetection(self.DETECTION_THRESHOLD)
            # # self._boxes[:,0,:] -= self.PADDING 
            # # self._boxes[:,1,:] += self.PADDING 
            # #self._resimg = self.vizDetections(img)
            # #cv2.imwrite(os.path.join(outDir, entry), self._resimg)
            # #self._resimg = SideObjectDetection.drawBBonImg(img, self._boxes[0])
            # #except:
            #     #print("No Object was found in picture")
            #     #self._resimg = None
            # self._resimg = self.compareToGood(img)
            try:
                self.analyseOneImg(img)
                #cv2.imwrite(os.path.join(outDir, entry), self._resimgBending)
            except RuntimeError as e : 
                print(f"couldn't find objects {e}")

        return
    
    def runRos (self) :
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown() :
            
            self.analyseOneImg(self._rgbImg)
            rate.sleep ()
        rospy.spin()
    
    def analyseOneImg(self, img):
        """!
        @brief analysis of one image

        Parameters : 
            @param self => object of the class
            @param img => np array of the RGB image
            @param cableType => type of the cable that is being analysed
            @param doBending => do the bending correction analysis
            @param doAngle => do th angle correction analysis

        """
        results =None
        
        self._resimgDetection = results.plot()
        self._boxes = results.boxes.xyxy.numpy().reshape((-1,2,2))
        self._predictions = results.boxes.conf.numpy()
        self._classes = results.boxes.cls.numpy()
        self.filterDetection(self.DETECTION_THRESHOLD)
        self._objDetImg = self.vizDetections(img)
        

        cv2.imshow("test", self._objDetImg)
        cv2.waitKey(1)
        # print ("Good")
        # print(self._boxesGoodBlack)
        # print("Meassured")
        # print(self._boxes)
        # print(self._classes)
        # print(self._predictions)
    
    
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
    
    def vizDetections(self,img):
        """!
        @brief visualises the detections stores in the objects attributes

        Parameters : 
            @param self => object of the class
            @param img => image to draw onto

        """
        img = img.copy()
        for index, box in enumerate(self._boxes):
            if (self._classes[index] == 0):
                img = TruckObjectDetection.drawBBonImg(img, box,(150,150,255))
            elif (self._classes[index] == 1):
                img = TruckObjectDetection.drawBBonImg(img, box,(150,255,150))
        return img

    def drawBBonImg(img, BB, color= (36,255,12)):
        """!
        @brief draws a bounding box in the format [[upper left x, upper left y], [lower right x, lower right y]] on th given image 

        Parameters : 
            @param img => image to draw onto
            @param BB => bounding box
            @param color = (36,255,12) => drawing color

        """
        start_point = BB[0].astype(int)
        end_point = BB[1].astype(int)
        thickness = 3
        image = cv2.rectangle(img, start_point, end_point, color, thickness) 
        return image
    
if __name__ == "__main__":
    rgb_topic = "/kinect1/color/image_raw"
    cable = TruckObjectDetection(MODEL_PATH,rgb_topic)
    cable.runRos()