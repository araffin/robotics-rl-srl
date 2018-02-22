import os
import json
from enum import Enum

import zmq

HOSTNAME = 'localhost'
SERVER_PORT = 7777


class Command(Enum):
    HELLO = 0
    LEARN = 1
    READY = 2
    ERROR = 3
    EXIT = 4


class SRLClient(object):
    def __init__(self, data_folder, hostname='localhost', server_port=7777):
        super(SRLClient, self).__init__()
        self.hostname = hostname
        self.server_port = server_port
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(hostname, server_port))
        self.path_to_srl_server = None
        self.data_folder = data_folder

    def __del__(self):
        self.socket.close()

    def waitForServer(self):
        print("Waiting for server...")
        msg = self.socket.recv_json()
        assert Command(msg['command']) == Command.HELLO
        self.path_to_srl_server = msg.get('path')
        self.socket.send_json({"command": Command.HELLO.value, 'data_folder': self.data_folder})
        print("Connected to server")

    def sendLearnCommand(self, state_dim, seed=1):
        self.socket.send_json({"command": Command.LEARN.value, 'state_dim': state_dim, 'seed': seed})

    def sendExitCommand(self):
        self.socket.send_json({"command": Command.EXIT.value})

    def receiveMessage(self):
        """
        :return: (Command, dict)
        """
        msg = self.socket.recv_json()
        try:
            # Convert to a command object
            command = Command(msg.get('command'))
        except ValueError:
            raise ValueError("Unknown command: {}".format(msg))
        return command, msg

    def waitForSRLModel(self, state_dim):
        """
        :param state_dim: (int)
        :return: (bool, str) (True if no error, path to learned model)
        """
        self.sendLearnCommand(state_dim)
        command, msg = self.receiveMessage()
        if command == Command.ERROR:
            print("An error occured during SRL")
            return False, ""
        elif command != Command.READY:
            print("Unsupported command:{}".format(command))
            return False, ""
        else:
            path_to_model = msg.get('path') + '/srl_model.pth'
        return True, path_to_model


if __name__ == '__main__':
    data_folder = 'test_server'
    os.makedirs('srl_priors/data/' + data_folder, exist_ok=True)

    dataset_config = {'relative_pos': False}
    with open("srl_priors/data/{}/dataset_config.json".format(data_folder), "w") as f:
        json.dump(dataset_config, f)

    socket_client = SRLClient(data_folder)
    socket_client.waitForServer()
    try:
        while True:
            ok, path_to_model = socket_client.waitForSRLModel(state_dim=3)
            print(path_to_model)
            break
    except KeyboardInterrupt:
        pass

    socket_client.sendExitCommand()
    print("Client exiting...")
