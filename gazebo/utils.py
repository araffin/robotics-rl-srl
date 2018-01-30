from __future__ import print_function, absolute_import, division

import sys

import zmq
import numpy as np

if sys.version_info > (3,):
    buffer = memoryview


def recvMatrix(socket):
    """
    Receive a numpy array over zmq
    :param socket: (zmq socket)
    :return: (Numpy matrix)
    """
    metadata = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    A = np.frombuffer(buffer(msg), dtype=metadata['dtype'])
    return A.reshape(metadata['shape'])


def sendMatrix(socket, mat):
    """
    Send a numpy mat with metadata over zmq
    :param socket:
    :param mat: (numpy matrix)
    """
    metadata = dict(
        dtype=str(mat.dtype),
        shape=mat.shape,
    )
    # SNDMORE flag specifies this is a multi-part message
    socket.send_json(metadata, flags=zmq.SNDMORE)
    return socket.send(mat, flags=0, copy=True, track=False)


def getActions(delta_pos, n_actions):
    """
    Get list of possible actions
    :param delta_pos: (float)
    :param n_actions: (int)
    :return: (numpy matrix)
    """
    possible_deltas = [i * delta_pos for i in range(-1, 2)]
    actions = []
    for dx in possible_deltas:
        for dy in possible_deltas:
            for dz in possible_deltas:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # Allow only move in one direction
                if abs(dx) + abs(dy) + abs(dz) > delta_pos:
                    continue
                actions.append([dx, dy, dz])

    assert len(actions) == n_actions, "Wrong number of actions: {}".format(len(actions))

    return np.array(actions)


def randomAction(possible_actions):
    """
    Take a random action for a list of possible actions
    :param possible_actions: [[float]
    :return: [float]
    """
    action_idx = np.random.randint(len(possible_actions))
    return possible_actions[action_idx]
