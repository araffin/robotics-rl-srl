#!/usr/bin/env python
"""
Teleoperation client:
- control the robobo with arrows keys + U for stopping the robot
- control the baxter robot with arrows keys + D and U keys (for moving along the z-axis)
Press esc or q to exit the client
"""
from __future__ import division, print_function, absolute_import

import time
import signal

import cv2
import numpy as np
import zmq

from .constants import SERVER_PORT, HOSTNAME, UP_KEY, DOWN_KEY, LEFT_KEY, \
    RIGHT_KEY, D_KEY, U_KEY, EXIT_KEYS, R_KEY, Move, IMAGE_TOPIC, DELTA_POS, USING_ROBOBO
from .utils import recvMatrix

np.set_printoptions(precision=4)

# Connect to the Gym bridge ROS node
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

print("Waiting for server...")
msg = socket.recv_json()
print("Connected to server")

times = []
action = [0, 0, 0]
if USING_ROBOBO:
    action_dict = {
        UP_KEY: Move.FORWARD.value,
        DOWN_KEY: Move.BACKWARD.value,
        LEFT_KEY: Move.LEFT.value,
        RIGHT_KEY: Move.RIGHT.value,
        D_KEY: Move.STOP.value,
        U_KEY: Move.STOP.value
    }
else:
    action_dict = {
        UP_KEY: [- DELTA_POS, 0, 0],
        DOWN_KEY: [DELTA_POS, 0, 0],
        LEFT_KEY: [0, - DELTA_POS, 0],
        RIGHT_KEY: [0, DELTA_POS, 0],
        D_KEY: [0, 0, - DELTA_POS],
        U_KEY: [0, 0, DELTA_POS]

    }

# Create dark image to listen to keyboard events
cv2.imshow("Image", np.zeros((100, 100, 3), dtype=np.uint8))

should_exit = [False]


# exit the script on ctrl+c
def ctrl_c(signum, frame):
    should_exit[0] = True


signal.signal(signal.SIGINT, ctrl_c)

while not should_exit[0]:
    # Retrieve pressed key
    key = cv2.waitKey(0) & 0xff

    if key in EXIT_KEYS:
        break
    elif key in action_dict.keys():
        action = action_dict[key]
        socket.send_json({"command": "action", "action": action})
    elif key == R_KEY:
        socket.send_json({"command": "reset"})
    else:
        print("Unknown key: {}".format(key))
        continue

    start_time = time.time()
    # Receive state data (position, etc)
    state_data = socket.recv_json()
    print('state data: {}'.format(state_data))

    if IMAGE_TOPIC is not None:
        # Receive a camera image from the server
        img = recvMatrix(socket)
        cv2.imshow("Image", img)

    times.append(time.time() - start_time)

socket.send_json({"command": "exit"})
cv2.destroyAllWindows()

print("Client exiting...")
print("{:.2f} FPS".format(len(times) / np.sum(times)))

socket.close()
