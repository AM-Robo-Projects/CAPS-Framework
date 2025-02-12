import cv2
import numpy as np
import pyrealsense2 as rs
import cv2.aruco as aruco
from tf.transformations import quaternion_about_axis, quaternion_matrix, quaternion_from_matrix
import time

class ArucoDetector:
    def __init__(self, marker_length, pub_live):
        self.marker_length = marker_length
        self.pub_live = pub_live
        self.poses = {}

        # Load RealSense camera pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # Initialize ArUco detector
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters_create()
        self.marker_points = np.array([[-self.marker_length / 2, self.marker_length / 2, 0],
                              [self.marker_length / 2, self.marker_length / 2, 0],
                              [self.marker_length / 2, -self.marker_length / 2, 0],
                              [-self.marker_length / 2, -self.marker_length / 2, 0]], dtype=np.float32)

        # Load camera intrinsics from RealSense
        self.intrinsics = None
        self.load_realsense_intrinsics()

    def load_realsense_intrinsics(self):
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(self.intrinsics.coeffs)

    def process_frame(self, frame):
        # Undistort frame
        frame_undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

        if np.all(ids is not None):
            for i in range(0, len(ids)):
                # SolvePnP with iterative method for better accuracy
                _, rvec, tvec = cv2.solvePnP(self.marker_points, corners[i], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                cv2.drawFrameAxes(frame_undistorted, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                # Construct the transformation matrix
                axis = rvec.flatten() / np.linalg.norm(rvec)
                angle = np.linalg.norm(rvec)
                q = quaternion_about_axis(angle, axis)
                tvec = tvec.flatten()

                # Create the transformation matrix (4x4) using quaternion and translation vector
                T = np.eye(4)
                T[:3, :3] = quaternion_matrix(q)[:3, :3]  # Rotation part
                T[:3, 3] = tvec  # Translation part

                # Store tag pose (in this case just as a dictionary)
                tag_pose = {
                    'position': {'x': tvec[0], 'y': tvec[1], 'z': tvec[2]},
                    'orientation': {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]},
                    'transformation_matrix': T
                }

                # For visualization or further processing, you can print the pose or transformation matrix
                print(f"Pose for marker {ids[i][0]}: {tag_pose}")
                print(f"Transformation Matrix for marker {ids[i][0]}:\n{T}")

    def run(self):
        try:
            while True:
                # Capture frame from RealSense camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert frame to numpy array
                frame = np.asanyarray(color_frame.get_data())
                self.process_frame(frame)

                # Display the frame
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Stop streaming when done
            self.pipeline.stop()

def main():
    marker_length = 0.1  # Marker size in meters
    pub_live = True  # Set to False if you don't want live broadcasting

    aruco_detector = ArucoDetector(marker_length, pub_live)
    aruco_detector.run()

if __name__ == '__main__':
    main()
