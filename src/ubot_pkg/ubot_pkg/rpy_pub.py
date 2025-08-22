import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2 as cv
from cv2 import aruco
import numpy as np

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.degrees([roll, pitch, yaw])

class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__("aruco_pose_publisher")
        self.bridge = CvBridge()

        # Subscribers and publishers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/aruco_pose', 10)

        # Load calibration
        calib_data_path = "MultiMatrix.npz"
        calib_data = np.load(calib_data_path)
        self.cam_mat = calib_data["camMatrix"]
        self.dist_coef = calib_data["distCoef"]

        # ArUco setup
        self.MARKER_SIZE = 8  # cm
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.param_markers = aruco.DetectorParameters()

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        marker_corners, marker_IDs, _ = aruco.detectMarkers(
            gray_frame,
            self.marker_dict,
            parameters=self.param_markers
        )

        if marker_IDs is not None:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners,
                self.MARKER_SIZE,
                self.cam_mat,
                self.dist_coef
            )

            for i, marker_id in enumerate(marker_IDs):
                # Rotation
                R_mat, _ = cv.Rodrigues(rVec[i])
                roll, pitch, yaw = rotationMatrixToEulerAngles(R_mat)

                # Translation
                x, y, z = tVec[i][0]

                # Publish as Float32MultiArray: [ID, x, y, z, roll, pitch, yaw]
                msg_out = Float32MultiArray()
                msg_out.data = [float(marker_id[0]), x, y, z, roll, pitch, yaw]
                self.publisher_.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
