
import subprocess
import numpy as np
import pickle

# log_dir = 'logs/OmnirobotEnv-v0/srl_combination/ppo2/19-04-24_10h36_52/'
# tasks=['cc']
# episodeEval(log_dir,tasks,save_name='episode_eval.npy',num_timesteps=300)
# for i in range(10):
#     time.sleep(1)
#     print(i)

#episodeEval(log_dir,tasks,save_name='episode_eval.npy',num_timesteps=300)

def dict2array(tasks,data):
    res=[]
    for t in tasks:
        res.append(data[t])
    res=np.array(res)
    return res

def episodeEval(log_dir, tasks,num_timesteps=1000,num_cpu=1):
    for t in tasks:
        eval_args=['--log-dir', log_dir, '--num-timesteps', str(num_timesteps), '--num-cpu',str(num_cpu)]
        task_args=['--task',t]

        subprocess.call(['python', '-m', 'rl_baselines.cross_eval_utils']+eval_args+task_args)
    file_name=log_dir+'episode_eval.pkl'
    with open(file_name, 'rb') as f:
        eval_reward = pickle.load(f)

    #Trasfer the data from dict into a numpy array and save
    eval_reward=dict2array(tasks,eval_reward)
    file_name=log_dir+'episode_eval.npy'
    np.save(file_name, eval_reward)

# if __name__ == '__main__':
#
#     log_dir = 'logs/OmnirobotEnv-v0/srl_combination/ppo2/19-04-24_10h36_52/'
#     tasks=['cc','sc','sqc']
#     episodeEval(log_dir, tasks, num_timesteps=800, num_cpu=1)

