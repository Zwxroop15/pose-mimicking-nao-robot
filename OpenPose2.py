# Import the necessary libraries
import json
import cv2 as cv
import numpy as np
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
cap = cv.VideoCapture(args.input if args.input else 0)

# Variables to lock onto the target body
target_keypoints = None

# Main loop to process each frame
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Preprocess the frame for the neural network
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Extract only the relevant data

    num_bodies = out.shape[0]  # Number of detected people
    detected_keypoints = []

    # Process all detected bodies
    for body_idx in range(num_bodies):
        points = []
        confidence_sum = 0

        for i in range(len(BODY_PARTS)):
            heatMap = out[body_idx, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            if conf > args.thr:
                points.append((int(x), int(y)))
                confidence_sum += conf
            else:
                points.append(None)

        detected_keypoints.append((confidence_sum, points))

    # Lock onto a target body if not already locked
    if target_keypoints is None and detected_keypoints:
        # Select the body with the highest confidence
        target_keypoints = max(detected_keypoints, key=lambda x: x[0])[1]

    if target_keypoints:
        # Match the target body in the current frame
        def compute_distance(target, candidate):
            return sum(np.linalg.norm(np.array(t) - np.array(c)) for t, c in zip(target, candidate) if t and c)

        # Find the closest match to the target
        best_match = None
        min_distance = float('inf')

        for _, keypoints in detected_keypoints:
            distance = compute_distance(target_keypoints, keypoints)
            if distance < min_distance:
                best_match = keypoints
                min_distance = distance

        # Update the target keypoints to the closest match
        if best_match:
            target_keypoints = best_match

        # Draw the skeleton for the target body only
        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if target_keypoints[idFrom] and target_keypoints[idTo]:
                cv.line(frame, target_keypoints[idFrom], target_keypoints[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, target_keypoints[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, target_keypoints[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        # Save joint data to JSON
        joint_data = {part: list(coord) for part, coord in zip(BODY_PARTS.keys(), target_keypoints) if coord}
        with open("pose_data.json", "w") as f:
            json.dump(joint_data, f)

    # Display the frame
    cv.imshow('OpenPose using OpenCV', frame)
