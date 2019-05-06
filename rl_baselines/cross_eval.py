
import subprocess
import numpy as np
import pickle
import argparse
import os

from rl_baselines.student_eval import  allPolicy
from srl_zoo.utils import printRed, printGreen
from rl_baselines.cross_eval_utils import EnvsKwargs, loadConfigAndSetup, policyEval,createEnv

def dict2array(tasks,data):
    res=[]
    print(data)
    for t in tasks:
        if(t=='sc'):
            max_reward=250
        else:
            max_reward=1850
        data[t]=data[t].astype(float)
        data[t][:,1:]=data[t][:,1:]/max_reward
        res.append(data[t])
    res=np.array(res)
    return res

def episodeEval(log_dir, tasks,num_timesteps=1000,num_cpu=1):
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




def policyCrossEval(log_dir,task,episode,model_path, num_timesteps=2000,num_cpu=1):
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

    log_dir, environment, algo_args = createEnv(log_dir, train_args, algo_name, algo_class, env_kwargs, num_cpu=num_cpu)

    reward = policyEval(environment, model_path, log_dir, algo_class, algo_args, num_timesteps, num_cpu)

    # Just a trick to save the episode number of the reward,but need a little bit more space to store
    reward = np.append(episode, reward)
    return reward, True


#
#
#
# def saveReward(log_dir,reward, task,save_name='episode_eval.pkl'):
#
#
#     file_name=log_dir+save_name
#
#     #can be changed accordingly
#     if(os.path.isfile(file_name)):
#
#
#         with open(file_name, 'rb') as f:
#             eval_reward= pickle.load(f)
#
#         if (task in eval_reward.keys()):
#             episodes = np.unique(eval_reward[task][:, 0])
#             #The fisrt dimension of reward is the episode
#             current_episode =reward[0]
#             #Check if the latest episodes policy is already saved
#             if (current_episode not in episodes):
#                 eval_reward[task]=np.append(eval_reward[task],[reward],axis=0)
#                 with open(file_name, 'wb') as f:
#                     pickle.dump(eval_reward, f, pickle.HIGHEST_PROTOCOL)
#         else:# The task is not in the file yet
#             eval_reward[task] =reward[None,:]
#             with open(file_name, 'wb') as f:
#                 pickle.dump(eval_reward, f, pickle.HIGHEST_PROTOCOL)
#     else: #There is still not a episodes rewards evaluation registered
#
#         eval_reward = {}
#
#         eval_reward[task]=reward[None,:]
#         with open(file_name, 'wb') as f:
#             pickle.dump(eval_reward, f, pickle.HIGHEST_PROTOCOL)
#
#     return

#python -m rl_baselines.cross_eval --log-dir logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-05-03_11h35_10/ --num-iteration 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation after training")
    parser.add_argument('--log-dir',type=str, default=''
                        ,help='RL algo to use')
    parser.add_argument('--num-iteration', type=int, default=5,
                        help='number of time each algorithm should be run the eval (N seeds).')
    args, unknown = parser.parse_known_args()


    log_dir =  args.log_dir
    #log_dir = 'logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-05-03_11h35_10/'

    tasks=['sc']
    episodes, policy_paths = allPolicy(log_dir)
    printRed(len(episodes))

    #
    #
    # episode_eval=[]
    #
    # for k in range(120):
    #     for i in range(10):
    #         print("Episode: {}".format(episodes[k]))
    #     for task_label in tasks:
    #         for seed_i in range(args.num_iteration):
    #             printRed(policy_paths[k])
    #             command=['python', '-m', 'rl_baselines.evaluation.eval_post', '--log-dir',log_dir,
    #                      '--task-label', task_label, '--episode', str(episodes[k]), '--policy-path' , policy_paths[k],
    #                      '--seed', str(seed_i)
    #                      ]
    #             subprocess.call(command)
    #
    # file_name = log_dir + 'episode_eval.pkl'
    # with open(file_name, 'rb') as f:
    #     eval_reward = pickle.load(f)
    #
    # # Trasfer the data from dict into a numpy array and save
    # data = eval_reward
    # for key in data.keys():
    #     data[key]=np.array(data[key][1])
    # printRed(data)
    # eval_reward = dict2array(tasks, data)
    # file_name = log_dir + 'episode_eval.npy'
    # np.save(file_name, eval_reward)

#######################################################


    task_label='cc'

    rewards = {}
    rewards['episode']=episodes[:186]
    rewards['policy']=policy_paths[:186]
    rewards[task_label] = []


    for k in range(186):
        model_path=policy_paths[k]

        local_reward = [int(episodes[k])]
        for seed_i in range(args.num_iteration):

            command_line_enjoy_student = ['python', '-m', 'replay.enjoy_baselines', '--num-timesteps', '251',
                                          '--log-dir', model_path,  "--seed", str(seed_i)]
            ok = subprocess.check_output(command_line_enjoy_student)
            ok = ok.decode('utf-8')
            str_before = "Mean reward: "
            str_after = "\npybullet"
            idx_before = ok.find(str_before) + len(str_before)
            idx_after = ok.find(str_after)
            seed_reward = float(ok[idx_before: idx_after])
            local_reward.append(seed_reward)
            print(local_reward)
        printRed("current rewards {} at episode {}".format(np.mean(local_reward), episodes[k]))
        rewards[task_label].append(local_reward)
        with open(args.log_dir+'/eval.pkl', 'wb') as f:
            pickle.dump(rewards, f, pickle.HIGHEST_PROTOCOL)


