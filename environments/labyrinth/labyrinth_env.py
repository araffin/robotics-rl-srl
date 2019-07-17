import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from environments.srl_env import SRLGymEnv
from state_representation.episode_saver import EpisodeSaver
from gym import spaces
#  Number of steps before termination
MAX_STEPS = 250  # WARNING: should be also change in __init__.py (timestep_limit)
# Terminate the episode if the arm is outside the safety sphere during too much time
RENDER_HEIGHT = 128
RENDER_WIDTH = 128
N_DISCRETE_ACTIONS = 4
MAZE_SIZE = 8

def getGlobals():
    """
    :return: (dict)
    """
    return globals()

class LabyrinthEnv(SRLGymEnv):
    """
    
    This Labyrinth environment could can at speed 28,000 FPS on 10 threads CPU (Intel i9-9900K)
    
    
    
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param random_target: (bool) Set the target to a random position
    :param state_dim: (int) When learning states
    
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    :param env_rank: (int) the number ID of the environment
    :param pipe: (Queue, [Queue]) contains the input and output of the SRL model

    :param srl_model: (str) The SRL_model used
    """

    def __init__(self, renders=False, is_discrete=True,
                 name="labyrinth", max_distance=1.6, shape_reward=False, record_data=False, srl_model="raw_pixels",
                 random_target=False, state_dim=-1, verbose=False, save_path='srl_zoo/data/', 
                 env_rank=0, srl_pipe=None, img_shape=None,  **_):
        super(LabyrinthEnv, self).__init__(srl_model=srl_model,
                                                relative_pos=False,
                                                env_rank=env_rank,
                                                srl_pipe=srl_pipe)
        self._env_step_counter = 0
        self._observation = []
        self._renders = renders
        self.img_shape = img_shape  # channel first !!
        if self.img_shape is None:
            self._width = RENDER_WIDTH
            self._height = RENDER_HEIGHT
        else:
            self._height, self._width = self.img_shape[1:]
        # self._shape_reward = shape_reward
        self._random_target = random_target
        self.terminated = False ## if already collected all tresors
        self.debug = False
        self.state_dim = state_dim
        self.saver = None
        self.verbose = verbose
        self.max_steps = MAX_STEPS
        self.robot_pos = np.zeros(2, dtype=int) ## index on the self.map array
        self.previous_robot_pos = None
        self.target_pos = []#np.zeros(2) # list of indexes on the self.map array
        
        
        # import ipdb; ipdb.set_trace()
        self.has_bumped = False  # Used for handling collisions
        self.count_collected_tresors = 0
        self.srl_model = srl_model
        self.maze_size = MAZE_SIZE
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        if record_data:
            self.saver = EpisodeSaver(name, max_distance, state_dim, globals_=getGlobals(), path=save_path)
        
        if self.srl_model == "ground_truth":
            self.state_dim = self.getGroundTruthDim()

        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
    def create_map(self):
        # Create map without robot
        ## put walls
        self.map = np.zeros((self.maze_size, self.maze_size)) ## -1 is wall, 0 is free space, 1 is key(tresor)
        self.map[:, 0] = -1
        self.map[:, -1] = -1
        self.map[0, :] = -1
        self.map[-1, :] = -1
        self.map[:3, 4] = -1
        self.map[-3:, 4] = -1
        ## put tresors
        self.target_pos.append(np.array([1, -2], dtype=int))
        self.target_pos.append(np.array([-2, 1], dtype=int))
        for key_pos in self.target_pos:
            self.map[key_pos[0], key_pos[1]] = 1
        valid_robot_pos_list = []
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if self.map[i, j] == 0:
                    valid_robot_pos_list.append(np.array([i, j], dtype=int))

        ### load images
        # self.background = 255*np.ones((self._height, self._width, 3), dtype=np.uint8)
        target_img_ori = cv2.imread("./environments/labyrinth/tresors_128.png")
        robot_img_ori = cv2.imread("./environments/labyrinth/corsair_128.png")
        map_h, map_w = self.map.shape
        self.square_size = int(self._height/map_h)
        self.target_img = cv2.resize(target_img_ori, (self.square_size, self.square_size))[..., ::-1]
        self.robot_img = cv2.resize(robot_img_ori, (self.square_size, self.square_size))[..., ::-1]
        
        return valid_robot_pos_list
    def getGroundTruthDim(self):
        """
        The new convention for the ground truth (GT): GT should include target position under random target
        setting. 
        """
        return self.maze_size**2

    def getTargetPos(self): ## TODO TODO TODO
        return np.hstack(self.target_pos)

    def getRobotPos(self): ## TODO TODO TODO
        return np.array(self.robot_pos)

    def getGroundTruth(self):
        """
        """
        # raise NotImplementedError
        return self.map.ravel()
        
    def reset(self):
        self.terminated = False
        valid_robot_pos_list = self.create_map()
        ## put robot
        self.robot_pos = valid_robot_pos_list[np.random.choice(np.arange(len(valid_robot_pos_list)))]
        self.map[self.robot_pos[0], self.robot_pos[1]] = 2
        
        self._env_step_counter = 0
        self._observation = self.getObservation(start=True)
        if self.saver is not None:
            self.saver.reset(self._observation, self.getTargetPos(), self.getRobotPos())
        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation)

        return self._observation

    def getObservation(self, start=False):
        """
        :return: (numpy array)
        """
        self._observation = self.render("rgb_array", start=start)
        return self._observation
    def valid_action(self, action):
        """
        check whether an action is valid
        return (bool)
        """
        if action == 0:
            real_action = np.array([1, 0])
        elif action == 1:
            real_action = np.array([0, 1])
        elif action == 2:
            real_action = np.array([-1, 0])
        elif action == 3:
            real_action = np.array([0, -1])
        else:
            raise NotImplementedError
        
        next_pos = self.robot_pos + real_action
        # import ipdb; ipdb.set_trace()
        valid = np.all(next_pos>=0) and np.all(next_pos<self.maze_size) and (self.map[next_pos[0], next_pos[1]] != -1)
        return valid, next_pos
    def step(self, action):
        ## action = 0, 1, 2, 3
        # True if it has bumped against a wall
        self._env_step_counter += 1
        self.has_bumped = False
        previous_pos = self.robot_pos.copy()
        self.previous_robot_pos = previous_pos
        valid, next_pos = self.valid_action(action)
        self.has_bumped = not valid
        reward = self._reward(next_pos) 
        done = self._termination() ## it's important to call self._reward before self.termination
        if self.has_bumped:
            ## no need to update observation
            pass
        else:
            self.map[previous_pos[0], previous_pos[1]] = 0 # free space
            ## update robot position and the map
            self.robot_pos = next_pos
            self.map[self.robot_pos[0], self.robot_pos[1]] = 2
            ## update observation
            self._observation = self.getObservation()

        if self.saver is not None:
            self.saver.step(self._observation, action, reward, done, self.getRobotPos())
        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation), reward, done, {}

        return self._observation, reward, done, {}

    def render(self, mode='rgb_array', start=False):
        ## [OPT] this part could be optimized again if needed, but it's not necessary, because 2,8000 FPS 
        ## has already been ~4 times faster then 7000 FPS of PPO2 (with 10 CPUs). 
        if self._observation is None or (isinstance(self._observation, list) and len(self._observation) == 0) or start:
            previous_obs = 255*np.ones((self._height, self._width, 3), dtype=np.uint8)
        else:
            previous_obs = self._observation
        # background = 255*np.ones((self._height, self._width, 3), dtype=np.uint8) ## Could be optimized ! (use previous obsevation, etc)
        # import ipdb; ipdb.set_trace()
        map_h, map_w = self.map.shape
        square_size = self.square_size
        for i in range(map_h):
            for j in range(map_w):
                if self.map[i, j] == -1:
                    previous_obs[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size, :] = 0
                elif self.map[i, j] == 1:
                    previous_obs[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size, :] = self.target_img
                elif self.map[i, j] == 2:
                    previous_obs[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size, :] = self.robot_img
                elif self.map[i, j] == 0:
                    # print(self.previous_robot_pos)
                    if self.previous_robot_pos is not None and i==self.previous_robot_pos[0] and j==self.previous_robot_pos[1]:
                        previous_obs[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size, :] = 255
                    else:
                        pass
                else:
                    raise NotImplementedError
        if self._renders:
            cv2.imshow("Keep pressing (any key) to display or 'q'/'Esc' to quit.", previous_obs)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27: ## 'q' or 'Esc'
                cv2.destroyAllWindows()
                raise KeyboardInterrupt
        return previous_obs#[..., ::-1]

    def _termination(self):
        """
        :return: (bool)
        """
        self.terminated = self.count_collected_tresors == len(self.target_pos)
        if self.terminated or self._env_step_counter > self.max_steps:
            self._observation = self.getObservation() ### ??? TODO
            return True
        return False

    def _reward(self, previous_pos):
        """
        :return: (float)
        """
        # Negative reward when it bumps into a wall
        if self.has_bumped:
            reward = -1
        elif (self.map[previous_pos[0], previous_pos[1]] == 1):
            self.count_collected_tresors += 1
            reward = 10
        elif self.terminated: 
            ## TODO, it's weird that the toolbox (RL part) doesn't stop counting reward after termination.
            reward = 0.1
        else:
            reward = -0.1
        
        return reward
    
    def interactive(self, show_map=False):
        image = self.getObservation(start=True)
        cv2.imshow("image", image[..., ::-1])
        while True:
            k = cv2.waitKey(0)
            if k == 27 or k == ord("q"): ## press 'Esc' or 'q' to quit
                break
            elif k == ord("2") or k == 84:
                ## down
                image, reward, done, _ = self.step(0)
                print(reward, done)
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            elif k == ord("6") or k == 83:
                ## right
                image, reward, done, _ = self.step(1)
                print(reward, done)
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            elif k == ord("8") or k == 82:
                ## up 
                image, reward, done, _ = self.step(2)
                print(reward, done)
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            elif k == ord("4") or k == 81:
                ## left
                image, reward, done, _ = self.step(3)
                print(reward, done)
                if show_map:
                    print(self.map)
                cv2.imshow("image", image[..., ::-1])
            else:
                print("You are pressing the key: {}".format(k))



if __name__ == "__main__":
    print("Start")
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Labyrinth environment (debug purpose)")
    parser.add_argument('--seed', default=0, type=int, help='random seed for initial robot position')
    parser.add_argument('--show-map', default=False, action='store_true', help='display map in terminal')
    args, unknown = parser.parse_known_args()
    np.random.seed(args.seed)
    Env = LabyrinthEnv()
    Env.reset()
    Env.interactive(show_map=args.show_map)
    

