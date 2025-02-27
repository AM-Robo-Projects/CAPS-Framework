import cv2
import numpy as np
import random




def drawBBonImg(self, img, BB, color=(36, 255, 12), class_name="Unknown"):
        
        start_point = (int(BB[0]), int(BB[1]))
        end_point = (int(BB[2]), int(BB[3]))
        
        thickness = 3

        # Draw the bounding box
        cv2.rectangle(img, start_point, end_point, color, thickness)

        # Draw the class name label above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(class_name, font, font_scale, font_thickness)[0]
        text_origin = (start_point[0], max(start_point[1] - 10, text_size[1] + 5))

        # Background rectangle for text
        cv2.rectangle(
            img,
            (text_origin[0], text_origin[1] - text_size[1] - 5),
            (text_origin[0] + text_size[0], text_origin[1] + 5),
            (0, 0, 0),
            -1,
        )
        
        # Write text on the image
        cv2.putText(img, class_name, (text_origin[0], text_origin[1]), font, font_scale, (255, 255, 255), font_thickness)

        return img

def getbbCenter(bb):
        """!
        @brief returns the center of a given bounding box in the format [[upper left x, upper left y], [lower right x, lower right y]]

        Parameters : 
            @param bb => array of the bounding box

        """
        x_center = (bb[0,0] + bb[1,0] )/ 2
        y_center = (bb[0,1] + bb[1,1] )/ 2
        return np.array([x_center, y_center])
 
