import cv2 as cv
from cv2 import aruco
import numpy as np

def rotationMatrixToEulerAngles(R):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
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


def main():
    # Path to the calibration file
    calib_data_path = r"/home/fayas/ubot_ws/src/ubot_pkg/ubot_pkg/MultiMatrix.npz"

    # Load camera calibration data
    calib_data = np.load(calib_data_path)
    print("Loaded Calibration Data:", calib_data.files)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]

    # Marker size in centimeters (or any unit used during calibration)
    MARKER_SIZE = 8  

    # Define the ArUco dictionary
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Define parameters for detection
    param_markers = aruco.DetectorParameters()

    # Initialize video capture (change 1 to 0 if using default webcam)
    cap = cv.VideoCapture(2)  # For IP Webcam, use the URL like: "http://<ip>:8080/video"

    if not cap.isOpened():
        print("Error: Cannot open video stream or webcam")
        exit()

    while True:
        # print("barani")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect ArUco markers
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        )

        if marker_IDs is not None:
            # Estimate pose of each detected marker
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )

            for i, ids in enumerate(marker_IDs):
                corners = marker_corners[i].reshape(4, 2).astype(int)

                # Draw marker boundaries
                cv.polylines(frame, [corners], True, (0, 255, 255), 2, cv.LINE_AA)

                # Get corner points
                top_left, top_right, bottom_right, bottom_left = corners

                # Calculate Euclidean distance from camera
                distance = np.linalg.norm(tVec[i][0])

                # Draw axis for pose visualization
                cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                
                R_mat, _ = cv.Rodrigues(rVec[i])

                # Convert rotation matrix to Euler angles (roll, pitch, yaw)
                roll, pitch, yaw = rotationMatrixToEulerAngles(R_mat)

                # Print or display them
                print(f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
                    
                # Display marker ID and distance
                cv.putText(frame,
                        f"ID: {ids[0]} Dist: {distance:.2f} cm",
                        tuple(top_right),
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (0, 0, 255),
                        2,
                        cv.LINE_AA)

                # Display X, Y, Z translation
                cv.putText(frame,
                        f"x:{tVec[i][0][0]:.1f} y:{tVec[i][0][1]:.1f} z:{tVec[i][0][2]:.1f}",
                        tuple(bottom_right),
                        cv.FONT_HERSHEY_PLAIN,
                        1.0,
                        (255, 0, 0),
                        2,
                        cv.LINE_AA)

        # Show video frame
        cv.imshow("ArUco Pose Estimation", frame)

        # Press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    main()