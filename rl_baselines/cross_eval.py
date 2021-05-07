
import subprocess
import numpy as np
import pickle
import argparse
import os

from rl_baselines.student_eval import  allPolicyFiles
from srl_zoo.utils import printRed, printGreen
from rl_baselines.evaluation.cross_eval_utils import EnvsKwargs, loadConfigAndSetup, policyEval,createEnv
from shutil import copyfile


def dict2array(tasks,data):
    """
    Convert the dictionary data set into a plotable array
    :param tasks: ['sc','cc'], the task key in the dictionary
    :param data: the dict itself
    :return:
    """
    res=[]
    for t in tasks:
        if(t!='cc'):
            max_reward=250
            min_reward = 0
        else:
            max_reward = 1920
            min_reward = 0

        data[t]=np.array(data[t]).astype(float)
        data[t][:,1:]=(data[t][:,1:]-min_reward)/max_reward
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
    """
    To do a cross evaluation for a certain policy for different tasks
    A version of real time evaluation but with some bugs to fix
    :param log_dir:
    :param task:
    :param episode:
    :param model_path:
    :param num_timesteps: How many timesteps to evaluate the policy
    :param num_cpu:
    :return:
    """
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


#Example commands:
#python -m rl_baselines.cross_eval --log-dir logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-05-03_11h35_10/ --num-iteration 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation after training")
    parser.add_argument('--log-dir',type=str, default=''
                        ,help='RL algo to use')
    parser.add_argument('--num-iteration', type=int, default=5,
                        help='number of time each algorithm should be run the eval (N seeds).')
    parser.add_argument('--scheduler',type = int, default=1,
                        help='A step scheduler for the evaluation')

    args, unknown = parser.parse_known_args()


    log_dir =  args.log_dir
    #log_dir = 'logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-05-03_11h35_10/'

    episodes, policy_paths = allPolicyFiles(log_dir)
    index_to_begin =0
    # The interval to skip, how many times we skip the evaluate
    # For example: if interval = 4 and episode, then the evaluation be be performed each 4* saved_checkpoint_episode
    interval_len = args.scheduler


    #To verify if the episodes have been evaluated before
    if(os.path.isfile(args.log_dir+'/eval.pkl')):
        with open(args.log_dir+'/eval.pkl', "rb") as file:
            rewards = pickle.load(file)

        max_eps = max(np.array(rewards['episode']).astype(int))
        index_to_begin = episodes.astype(int).tolist().index(max_eps)+1


    else:
        task_labels = ['cc', 'sc','esc']
        rewards = {}
        rewards['episode'] = []
        rewards['policy'] = []
        for t in ['cc', 'sc','esc']:
            rewards[t] = []


    for policy_path in policy_paths[index_to_begin:]:
        copyfile(log_dir+'/args.json', policy_path+'/args.json')
        copyfile(log_dir + '/env_globals.json', policy_path + '/env_globals.json')

    printGreen("The evaluation will begin from {}".format(episodes[index_to_begin]))

    last_mean = [250.,250,1900]
    run_mean = [0,0,0]


    for k in range(index_to_begin, len(episodes) ,interval_len):
        # if(interval_len > 1 and int(episodes[k])>=episode_schedule):
        #     k += interval_len-1
        printGreen("Evaluation for episode: {}".format(episodes[k]))
        increase_interval = True

        model_path=policy_paths[k]

        for t , task_label in enumerate(["-esc","-sc", "-cc" ]):

            local_reward = [int(episodes[k])]

            for seed_i in range(args.num_iteration):
                command_line_enjoy_student = ['python', '-m', 'replay.enjoy_baselines', '--num-timesteps', '251',
                                              '--log-dir', model_path, task_label,  "--seed", str(seed_i)]
                ok = subprocess.check_output(command_line_enjoy_student)
                ok = ok.decode('utf-8')
                str_before = "Mean reward: "
                str_after = "\npybullet"
                idx_before = ok.find(str_before) + len(str_before)
                idx_after = ok.find(str_after)
                seed_reward = float(ok[idx_before: idx_after])

                local_reward.append(seed_reward)
                printGreen(local_reward)
                printRed("current rewards {} at episode {} with random seed: {} for task {}".format(
                np.mean(seed_reward), episodes[k], seed_i, task_label))

            rewards[task_label[1:]].append(local_reward)
            run_mean[t] = np.mean(local_reward[1:])



        # If one of the two mean rewards varies more thant 1%, then we do not increase the evaluation interval
        for t in range(len(run_mean)):
            if(run_mean[t] > 1.01 * last_mean[t] or run_mean[t] <0.99 *last_mean[t]):
                increase_interval = False

        printGreen("Reward now: {}, last Rewards: {} for sc and cc respectively".format(run_mean, last_mean))
        # If the mean reward varies slowly, we increase the length of the evaluation interval
        if (increase_interval):
            current_eps = episodes[k]
            k = k + 5
            printGreen("Reward at current episode {} varies slow, change to episode {} for next evaluation"
                       .format(current_eps, episodes[k]))

        last_mean = run_mean.copy()

        rewards['episode'].append(int(episodes[k]))
        rewards['policy'].append(model_path)
        with open(args.log_dir+'/eval.pkl', 'wb') as f:
            pickle.dump(rewards, f, pickle.HIGHEST_PROTOCOL)


