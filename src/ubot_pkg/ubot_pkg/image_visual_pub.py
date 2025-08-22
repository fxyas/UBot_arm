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

class ArucoVisualizer(Node):
    def __init__(self):
        super().__init__("aruco_visualizer")
        self.bridge = CvBridge()

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_plot', 10)
        self.pose_pub = self.create_publisher(Float32MultiArray, '/aruco_pose', 10)

        # Load calibration data
        calib_data_path = "/home/fayas/ubot_ws/src/ubot_pkg/ubot_pkg/MultiMatrix.npz"
        calib_data = np.load(calib_data_path)
        self.cam_mat = calib_data["camMatrix"]
        self.dist_coef = calib_data["distCoef"]

        # ArUco setup
        self.MARKER_SIZE = 8  # cm
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.param_markers = aruco.DetectorParameters()

        self.get_logger().info("Aruco Visualizer Node Started!")

    def image_callback(self, msg):
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
                corners = marker_corners[i].reshape(4, 2).astype(int)
                top_left, top_right, bottom_right, bottom_left = corners

                # Draw marker boundaries and axes
                cv.polylines(frame, [corners], True, (0, 255, 255), 2, cv.LINE_AA)
                cv.drawFrameAxes(frame, self.cam_mat, self.dist_coef, rVec[i], tVec[i], 4, 4)

                # Rotation matrix to Euler
                R_mat, _ = cv.Rodrigues(rVec[i])
                roll, pitch, yaw = rotationMatrixToEulerAngles(R_mat)

                # Translation
                x, y, z = tVec[i][0]

                # Display on image
                distance = np.linalg.norm(tVec[i][0])
                cv.putText(frame,
                           f"ID:{marker_id[0]} Dist:{distance:.2f} cm",
                           tuple(top_right),
                           cv.FONT_HERSHEY_PLAIN,
                           1.3,
                           (0, 0, 255),
                           2,
                           cv.LINE_AA)
                cv.putText(frame,
                           f"x:{x:.1f} y:{y:.1f} z:{z:.1f}",
                           tuple(bottom_right),
                           cv.FONT_HERSHEY_PLAIN,
                           1.0,
                           (255, 0, 0),
                           2,
                           cv.LINE_AA)
                cv.putText(frame,
                           f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}",
                           (10, 30 + i*20),
                           cv.FONT_HERSHEY_PLAIN,
                           1.0,
                           (0, 255, 0),
                           2,
                           cv.LINE_AA)

                # Publish pose data
                pose_msg = Float32MultiArray()
                pose_msg.data = [float(marker_id[0]), x, y, z, roll, pitch, yaw]
                self.pose_pub.publish(pose_msg)

        # Publish image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(img_msg)
        self.get_logger().info("Publishing image with markers")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
