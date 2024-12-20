from naoqi import ALProxy
import time

# Define the robot's IP and port
robot_ip = "127.0.0.1"  # Localhost (for virtual robot)
robot_port = 56922  # Port used by the virtual robot

# Connect to ALMotion and ALRobotPosture services
motion = ALProxy("ALMotion", robot_ip, robot_port)
posture = ALProxy("ALRobotPosture", robot_ip, robot_port)

# Test 1: Move the robot's head pitch (up and down)
print("Moving head pitch...")
motion.setAngles("HeadPitch", 0.5, 0.2)  # Move head up
time.sleep(2)
motion.setAngles("HeadPitch", -0.5, 0.2)  # Move head down
time.sleep(2)

# Test 2: Move the robot's head yaw (left and right)
print("Moving head yaw...")
motion.setAngles("HeadYaw", 0.5, 0.2)  # Turn head right
time.sleep(2)
motion.setAngles("HeadYaw", -0.5, 0.2)  # Turn head left
time.sleep(2)

# Test 3: Make the robot wave its right arm
print("Making robot wave...")
motion.setAngles("RShoulderPitch", 0.5, 0.5)  # Move shoulder pitch
motion.setAngles("RElbowYaw", -1.5, 0.5)  # Move elbow yaw
motion.setAngles("RElbowRoll", -0.5, 0.5)  # Move elbow roll
motion.setAngles("RWristYaw", 1.0, 0.5)   # Move wrist yaw
time.sleep(2)

# Return to neutral posture after waving
motion.setAngles("RShoulderPitch", 1.5, 0.5)
motion.setAngles("RElbowYaw", 0.0, 0.5)
motion.setAngles("RElbowRoll", 0.0, 0.5)
motion.setAngles("RWristYaw", 0.0, 0.5)
time.sleep(2)

# Test 4: Make the robot sit down again
print("Making robot sit down...")
posture.goToPosture("Sit", 0.5)

print("Test complete!")
