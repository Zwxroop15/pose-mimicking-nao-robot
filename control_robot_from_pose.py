import cv2 as cv
import numpy as np
import argparse
from naoqi import ALProxy
from collections import deque
import time
import threading

# NAOqi Configuration
NAO_IP = "127.0.0.1"  # IP of the robot
NAO_PORT = 56855 # Port for robot communication

motion = ALProxy("ALMotion", NAO_IP, NAO_PORT)
posture = ALProxy("ALRobotPosture", NAO_IP, NAO_PORT)

# OpenPose Body Parts
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

NAO_PARTS = {
    "LShoulderPitch": ["Neck", "LShoulder"],
    "LShoulderRoll": ["LShoulder", "LElbow"],
    "LElbowYaw": ["LShoulder", "LElbow"],
    "LElbowRoll": ["LElbow", "LWrist"],
    "RShoulderPitch": ["Neck", "RShoulder"],
    "RShoulderRoll": ["RShoulder", "RElbow"],
    "RElbowYaw": ["RShoulder", "RElbow"],
    "RElbowRoll": ["RElbow", "RWrist"],
    "LHipPitch": ["Neck", "LHip"],
    "RHipPitch": ["Neck", "RHip"]
}

angle_buffers = {part: deque(maxlen=5) for part in NAO_PARTS.keys()}

def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.arctan2(dy, dx)
    return np.clip(angle, -1.5, 1.5)

def smooth_angle(part, angle):
    angle_buffers[part].append(angle)
    return np.mean(angle_buffers[part])

def setup_robot():
    posture.goToPosture("StandInit", 0.5)
    motion.wbEnable(True)
    motion.setStiffnesses("Body", 1.0)

def process_frame(frame, net, args):
    frame_resized = cv.resize(frame, (320, 240))  # Reduced resolution

    inp_blob = cv.dnn.blobFromImage(frame_resized, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(inp_blob)
    out = net.forward()

    points = [None] * len(BODY_PARTS)

    for i in range(len(BODY_PARTS)):
        heat_map = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heat_map)
        x = int(frame_resized.shape[1] * point[0] / out.shape[3])
        y = int(frame_resized.shape[0] * point[1] / out.shape[2])
        points[i] = (x, y) if conf > args.thr else None

    return frame_resized, points

def control_robot(points):
    for nao_part, (part_from, part_to) in NAO_PARTS.items():
        if points[BODY_PARTS[part_from]] and points[BODY_PARTS[part_to]]:
            angle = calculate_angle(points[BODY_PARTS[part_from]], points[BODY_PARTS[part_to]])
            smoothed_angle = smooth_angle(nao_part, angle)
            motion.setAngles(nao_part, smoothed_angle, 0.2)  # Smooth movement

def balance_reset():
    motion.setStiffnesses("Legs", 1.0)
    motion.wbEnable(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to video file or leave blank for webcam')
    parser.add_argument('--thr', default=0.2, type=float, help='Confidence threshold for pose detection')
    args = parser.parse_args()

    cap = cv.VideoCapture(args.input if args.input else 0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)  # Reduced width
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced height

    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    # Try to set preferable backend and target for optimal performance
    try:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    except AttributeError:
        print("Warning: OpenCV version does not support backend/target settings. Using defaults.")

    setup_robot()
    balance_reset()

    frame_counter = 0
    last_control_time = time.time()  # For limiting the number of control updates per second
    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            break

        # Process every frame and control the robot every 100ms (10 FPS)
        frame_resized, points = process_frame(frame, net, args)

        if time.time() - last_control_time >= 0.1:  # Only process control update every 100ms
            if points:
                control_robot(points)
            balance_reset()
            last_control_time = time.time()

        # Draw the pose estimation on the frame
        if frame_resized is not None and points:
            for pair in POSE_PAIRS:
                part_from, part_to = pair
                id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]
                if points[id_from] and points[id_to]:
                    cv.line(frame_resized, points[id_from], points[id_to], (0, 255, 0), 3)
                    cv.ellipse(frame_resized, points[id_from], (5, 5), 0, 0, 360, (0, 0, 255), cv.FILLED)
                    cv.ellipse(frame_resized, points[id_to], (5, 5), 0, 0, 360, (0, 0, 255), cv.FILLED)

            cv.imshow('Robot Mimicry', frame_resized)

        if cv.waitKey(10) == 27:
            break

        frame_counter += 1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
