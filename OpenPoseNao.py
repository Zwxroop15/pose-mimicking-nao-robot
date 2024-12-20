import cv2
#import mediapipe as mp
import numpy as np
from naoqi import ALProxy

# NAOqi Robot Configuration
NAO_IP = "127.0.0.1"
NAO_PORT = 50003
motion = ALProxy("ALMotion", NAO_IP, NAO_PORT)
posture = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)

# Initialize MediaPipe Pose
#mp_pose = mp.solutions.pose
#pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define NAO joint mappings
BODY_PARTS = {
    "LShoulderPitch": [11, 13],  # Left Hip to Left Knee
    "RShoulderPitch": [12, 14],  # Right Hip to Right Knee
    "LElbowRoll": [13, 15],      # Left Knee to Left Ankle
    "RElbowRoll": [14, 16],      # Right Knee to Right Ankle
}

def calculate_angle(p1, p2):
    """Calculate the angle between two points."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.arctan2(dy, dx)
    return np.clip(angle, -1.5, 1.5)

def setup_robot():
    """Set up the robot in an initial posture."""
    posture.goToPosture("StandInit", 0.5)
    motion.wbEnable(True)
    motion.setStiffnesses("Body", 1.0)

def control_robot(landmarks):
    """Control the robot using MediaPipe landmarks."""
    frame_height, frame_width = landmarks.shape
    for nao_joint, (from_idx, to_idx) in BODY_PARTS.items():
        if landmarks[from_idx] is not None and landmarks[to_idx] is not None:
            p1 = landmarks[from_idx]
            p2 = landmarks[to_idx]
            angle = calculate_angle(p1, p2)
            motion.setAngles(nao_joint, angle, 0.2)

def main():
    cap = cv2.VideoCapture(0)  # Use webcam or replace with a video file path
    setup_robot()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmarks.append((x, y))
            landmarks = np.array(landmarks)

            # Control robot based on landmarks
            control_robot(landmarks)

            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Display the output
        cv2.imshow("MediaPipe Pose with NAO Control", frame)
        if cv2.waitKey(10) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
