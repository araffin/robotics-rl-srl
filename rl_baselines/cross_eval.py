"""
Modified version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/visualize.py
Script used to send plot data to visdom
"""
import glob
import os
import json
import numpy as np
import tensorflow as tf
from scipy.signal import medfilt
from rl_baselines.utils import WrapFrameStack,computeMeanReward
from rl_baselines import AlgoType
from rl_baselines.registry import registered_rl
from srl_zoo.utils import printYellow, printGreen,printBlue,printRed
from datetime import datetime

def loadConfigAndSetup(log_dir):
    algo_name = ""
    for algo in list(registered_rl.keys()):
        if algo in log_dir:
            algo_name = algo
            break
    algo_class, algo_type, _ = registered_rl[algo_name]
    if algo_type == AlgoType.OTHER:
        raise ValueError(algo_name + " is not supported for evaluation")

    env_globals = json.load(open(log_dir + "env_globals.json", 'r'))
    train_args = json.load(open(log_dir + "args.json", 'r'))
    env_kwargs = {
        "renders": False,
        "shape_reward": False,  #TODO, since we dont use simple target, we should elimanate this choice?
        "action_joints": train_args["action_joints"],
        "is_discrete": not train_args["continuous_actions"],
        "random_target": train_args.get('random_target', False),
        "srl_model": train_args["srl_model"]
    }

    # load it, if it was defined
    if "action_repeat" in env_globals:
        env_kwargs["action_repeat"] = env_globals['action_repeat']

    # Remove up action
    if train_args["env"] == "Kuka2ButtonGymEnv-v0":
        env_kwargs["force_down"] = env_globals.get('force_down', True)
    else:
        env_kwargs["force_down"] = env_globals.get('force_down', False)

    if train_args["env"] == "OmnirobotEnv-v0":
        env_kwargs["simple_continual_target"] = env_globals.get("simple_continual_target", False)
        env_kwargs["circular_continual_move"] = env_globals.get("circular_continual_move", False)
        env_kwargs["square_continual_move"] = env_globals.get("square_continual_move", False)
        env_kwargs["eight_continual_move"] = env_globals.get("eight_continual_move", False)

    srl_model_path = None
    if train_args["srl_model"] != "raw_pixels":
        train_args["policy"] = "mlp"
        path = env_globals.get('srl_model_path')

        if path is not None:
            env_kwargs["use_srl"] = True
            # Check that the srl saved model exists on the disk
            assert os.path.isfile(env_globals['srl_model_path']), "{} does not exist".format(env_globals['srl_model_path'])
            srl_model_path = env_globals['srl_model_path']
            env_kwargs["srl_model_path"] = srl_model_path

    return train_args, algo_name, algo_class, srl_model_path, env_kwargs

def listEnvsKwargs(tasks,env_kwargs):

    tasks_env_kwargs=[]
    tmp=env_kwargs.copy()
    tmp['simple_continual_target'] = False
    tmp['circular_continual_move'] = False
    tmp['square_continual_move']   = False
    tmp['eight_continual_move']    = False
    for t in tasks:
        #For every tasks we create a special env_kwargs to generate different envs
        if (t=='sc'):
            tmp['simple_continual_target']=True
            tasks_env_kwargs.append(tmp.copy())
            tmp['simple_continual_target']=False
        elif (t=='cc'):

            tmp['circular_continual_move']=True
            tasks_env_kwargs.append(tmp.copy())
            tmp['circular_continual_move']=False
        elif (t=='sqc'):
            tmp['square_continual_move']=True
            tasks_env_kwargs.append(tmp.copy())
            tmp['square_continual_move']=False
        elif (t=='ec'):
            tmp['eight_continual_move']=True
            tasks_env_kwargs.append(tmp.copy())
            tmp['eight_continual_move']=False
    return tasks_env_kwargs

def createEnv( model_dir,train_args, algo_name, algo_class, env_kwargs, log_dir="/tmp/gym/test/",num_cpu=1,seed=0):

    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo_name, datetime.now().strftime("%y-%m-%d_%Hh%M_%S_%f"))
    os.makedirs(log_dir, exist_ok=True)
    args = {
        "env": train_args['env'],
        "seed":seed,
        "num_cpu": num_cpu,
        "num_stack": train_args["num_stack"],
        "srl_model": train_args["srl_model"],
        "algo_type": train_args.get('algo_type', None),
        "log_dir": log_dir
    }
    algo_args = type('attrib_dict', (), args)()  # anonymous class so the dict looks like Arguments object
    envs = algo_class.makeEnv(algo_args, env_kwargs=env_kwargs, load_path_normalise=model_dir)

    return log_dir, envs, algo_args




def policyEval(envs,model_path,log_dir,algo_class,algo_args,num_timesteps=1000,num_cpu=1):
    tf.reset_default_graph()
    method = algo_class.load(model_path, args=algo_args)
    using_custom_vec_env = isinstance(envs, WrapFrameStack)
    obs = envs.reset()
    if using_custom_vec_env:
        obs = obs.reshape((1,) + obs.shape)
    n_done = 0
    last_n_done = 0
    episode_reward=[]
    dones = [False for _ in range(num_cpu)]

    for i in range(num_timesteps):
        actions=method.getAction(obs,dones)
        obs, rewards, dones, _ = envs.step(actions)
        if using_custom_vec_env:
            obs = obs.reshape((1,) + obs.shape)
        if using_custom_vec_env:
            if dones:
                obs = envs.reset()
                obs = obs.reshape((1,) + obs.shape)

        n_done += np.sum(dones)
        if (n_done - last_n_done) > 1:
            last_n_done = n_done
            _, mean_reward = computeMeanReward(log_dir, n_done)
            episode_reward.append(mean_reward)

    _, mean_reward = computeMeanReward(log_dir, n_done)

    episode_reward.append(mean_reward)

    episode_reward=np.array(episode_reward)
    return episode_reward




def printEnvTasks(list_env_kwargs,tasks):
    """
    A Debugger to  verify the env_kwargs have the tasks that we want
    :param list_env_kwargs:
    :param tasks:
    :return:
    """
    i=0
    for env_kwargs in list_env_kwargs:

        print("For env kwargs {} and task: {}".format(i,tasks[i]))
        print("sc :",env_kwargs['simple_continual_target'])
        print("ec :", env_kwargs['eight_continual_move'])
        print("sqc:", env_kwargs['square_continual_move'])
        print("cc :", env_kwargs['circular_continual_move'])
        i+=1


def listEnvs(tasks,env_kwargs,train_args, algo_name, algo_class,model_dir):
    # For different tasks, we create a list of envs_kwargs to create different envs
    list_envs_kwargs = listEnvsKwargs(tasks, env_kwargs)
    log_dirs = []
    environments = []
    algo_args_list = []
    for kwargs in list_envs_kwargs:
        log_dir, envs, algo_args = createEnv( model_dir, train_args, algo_name, algo_class, kwargs)
        log_dirs.append(log_dir)
        environments.append(envs)
        algo_args_list.append(algo_args)
    return (log_dirs,environments,algo_args_list)


def latestPolicy(log_dir,algo_name):
    files= glob.glob(os.path.join(log_dir+algo_name+'_*_model.pkl'))
    files_list = []
    for file in files:
        eps=int((file.split('_')[-2]))
        files_list.append((eps,file))

    def sortFirst(val):
        return val[0]

    files_list.sort(key=sortFirst)
    if len(files_list)>0:
        #episode,latest model file path, OK
        return files_list[-1][0],files_list[-1][1],True
    else:
        #No model saved yet
        return 0,'',False

def policyCrossEval(log_dir,tasks,num_timesteps=2000):


    train_args, algo_name, algo_class, srl_model_path, env_kwargs = loadConfigAndSetup(log_dir)

    episode, model_path,OK=latestPolicy(log_dir,algo_name)
    if(not OK):
        #no latest model saved yet
        return None, None, False
    else:
        OK=True
    printGreen("Evaluation from the model saved at: {}, with evaluation time steps: {}".format(model_path, num_timesteps))

    log_dirs, environments, algo_args_list = listEnvs( tasks=tasks, env_kwargs=env_kwargs,
                                                      train_args=train_args, algo_name=algo_name, algo_class=algo_class,
                                                      model_dir=log_dir)
    rewards=[]

    for i in range(len(tasks)):
        rewards.append(policyEval(environments[i], model_path, log_dirs[i],  algo_class, algo_args_list[i], num_timesteps))


    #Just a trick to save the episode number of the reward,but need a little bit more space to store
    tmp=rewards[-1].copy()
    rewards.append(tmp)
    rewards=np.array(rewards)
    rewards[-1]=episode

    return rewards, OK


'''
The most important function
'''
def episodeEval(log_dir,tasks,save_name='episode_eval.npy'):
    #log_dir='logs/OmnirobotEnv-v0/ground_truth/ppo2/19-04-19_14h35_31/'

    file_name=log_dir+save_name

    num_timesteps=1000
    if(os.path.isfile(file_name)):
        #print(file_name)
        eval_reward=np.load(file_name)
        # eval_reward: (np.array) [times of evaluation,number of tasks+1,number of episodes in one evaluation ]
        episodes=np.unique(eval_reward[:, -1,: ])
        printRed(episodes)
        rewards, ok = policyCrossEval(log_dir, tasks, num_timesteps)
        # rewards shape: [number of tasks+1, number of episodes in one evaluation]

        if (ok):
            current_episode =np.unique(rewards[-1,:])[0]
            #Check if the latest episodes policy is already saved
            if (current_episode not in episodes):
                eval_reward=np.append(eval_reward,[rewards],axis=0)
                np.save(file_name, eval_reward)
    else: #There is still not a episodes rewards evaluation registered
        rewards, ok = policyCrossEval(log_dir, tasks, num_timesteps)
        eval_reward = []
        if(ok):
            eval_reward.append(rewards)
        eval_reward=np.array(eval_reward)
        np.save(file_name, eval_reward)
    return
