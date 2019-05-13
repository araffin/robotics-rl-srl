
import subprocess
import numpy as np
import pickle
import argparse
import os

from rl_baselines.student_eval import  allPolicy
from srl_zoo.utils import printRed, printGreen
from rl_baselines.evaluation.cross_eval_utils import EnvsKwargs, loadConfigAndSetup, policyEval,createEnv

def dict2array(tasks,data):
    res=[]

    for t in tasks:
        if(t=='sc'):
            max_reward=250
        else:
            max_reward=1850

        data[t][:,1:]=data[t][:,1:]/max_reward
        res.append(data[t])
    res=np.array(res)
    return res

def episodeEval(log_dir, tasks,num_timesteps=1000):
    for t in tasks:
        eval_args=['--log-dir', log_dir, '--num-timesteps', str(num_timesteps), '--num-cpu',str(5)]
        task_args=['--task',t]

        subprocess.call(['python', '-m', 'rl_baselines.cross_eval_utils']+eval_args+task_args)
    file_name=log_dir+'episode_eval.pkl'
    with open(file_name, 'rb') as f:
        eval_reward = pickle.load(f)

    #Trasfer the data from dict into a numpy array and save
    eval_reward=dict2array(tasks,eval_reward)
    file_name=log_dir+'episode_eval.npy'
    np.save(file_name, eval_reward)




def policyCrossEval(log_dir,task,episode,model_path, num_timesteps=2000,num_cpu=1,seed=0):
    train_args, algo_name, algo_class, srl_model_path, env_kwargs = loadConfigAndSetup(log_dir)
    env_kwargs = EnvsKwargs(task, env_kwargs)

    OK = True
    if (not OK):
        # no latest model saved yet
        return None, False
    else:
        pass
    printGreen(
        "Evaluation from the model saved at: {}, with evaluation time steps: {}".format(model_path, num_timesteps))

    log_dir, environment, algo_args = createEnv(log_dir, train_args, algo_name, algo_class, env_kwargs, num_cpu=num_cpu,seed=seed)

    reward = policyEval(environment, model_path, log_dir, algo_class, algo_args, num_timesteps, num_cpu)

    # Just a trick to save the episode number of the reward,but need a little bit more space to store
    reward = np.append(episode, reward)
    return reward, True





def saveReward(log_dir,reward, task,save_name='episode_eval.pkl'):
    reward = reward.astype(float)

    file_name=log_dir+save_name

    #can be changed accordingly
    if(os.path.isfile(file_name)):


        with open(file_name, 'rb') as f:
            eval_reward= pickle.load(f)

        if (task in eval_reward.keys()):
            episodes = eval_reward[task][0]
            #The fisrt dimension of reward is the episode
            current_episode =reward[0]
            #Check if the latest episodes policy is already saved
            if (current_episode not in episodes):
            #     # eval_reward[task]=np.append(eval_reward[task],[reward],axis=0)
                eval_reward[task][0].append(reward[0])
                eval_reward[task][1].append(reward.tolist())
            else:
                index = episodes.index(current_episode)
                eval_reward[task][1][index].extend(reward[1:])
            with open(file_name, 'wb') as f:
                pickle.dump(eval_reward, f, pickle.HIGHEST_PROTOCOL)
        else:# The task is not in the file yet
            eval_reward[task]=([reward[0]],[reward.tolist()])
            with open(file_name, 'wb') as f:
                pickle.dump(eval_reward, f, pickle.HIGHEST_PROTOCOL)
    else: #There is still not a episodes rewards evaluation registered

        eval_reward = {}
        eval_reward[task]=([reward[0]],[reward.tolist()])
        with open(file_name, 'wb') as f:
            pickle.dump(eval_reward, f, pickle.HIGHEST_PROTOCOL)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation after training")
    parser.add_argument('--log-dir',type=str, default=''
                        ,help='RL algo to use')
    parser.add_argument('--task-label', type=str, default='',
                        help='task to evaluate')
    parser.add_argument('--episode', type=str, default='',
                        help='evaluation for the policy saved at this episode')
    parser.add_argument('--policy-path', type=str, default='',
                        help='policy path')
    parser.add_argument('--seed', type=int, default=0,
                        help='policy path')
    args, unknown = parser.parse_known_args()




    reward, _ = policyCrossEval(args.log_dir, args.task_label, episode=args.episode, model_path=args.policy_path,
                                num_timesteps=251,seed=args.seed)
    saveReward(args.log_dir, reward, args.task_label, save_name='episode_eval.pkl')





