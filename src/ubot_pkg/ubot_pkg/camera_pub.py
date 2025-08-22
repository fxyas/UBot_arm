import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__("camera_publisher")
        self.publisher = self.create_publisher(Image,"camera/image_raw",10)
        self.cap = cv2.VideoCapture(2)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.03,self.publish_frame)

    def publish_frame(self):
        ret,frame = self.cap.read()
        if not ret:
            self.get_logger().warn("camera frame not captured")
            return
        msg = self.bridge.cv2_to_imgmsg(frame,encoding="bgr8")
        self.publisher.publish(msg)
        self.get_logger().info("publishing frame")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()