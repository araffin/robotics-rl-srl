"""
Modified version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/visualize.py
Script used to send plot data to visdom
"""
import glob
import os
import json
import numpy as np
import tensorflow as tf
import pickle
from srl_zoo.utils import printRed
from rl_baselines.utils import WrapFrameStack,computeMeanReward
from stable_baselines.common import set_global_seeds
from rl_baselines import AlgoType
from rl_baselines.registry import registered_rl
from rl_baselines.cross_eval_utils import  loadConfigAndSetup, latestPolicy
from datetime import datetime
import subprocess

def newFile(episode,filename):
    #Verify if the new saved policy is already learned by the student
    return

def srl_train(teacher_path):
    tmp_dir=os.getcwd()
    os.chdir('srl_zoo')
    # python - m
    # environments.dataset_generator - -num - cpu
    # 6 - -name
    # Omnibot_random_simple - -env
    # OmnirobotEnv - v0 - -simple - continual - -num - episode
    # 250 - f
    printRed(os.getcwd())
    subprocess.call(['python '])
    os.chdir(tmp_dir)
    printRed(os.getcwd())
    return



def DatasetGenerator(teacher_path,output_name,task_id,episode=-1,
                            env_name='OmnirobotEnv-v0',  num_cpu=1,num_eps=200):
    command_line = ['python','-m', 'environments.dataset_generator_student','--run-policy', 'custom']
    cpu_command  = ['--num-cpu',str(num_cpu)]
    name_command = ['--name',output_name]
    env_command  = ['--env',env_name]
    task_command = [task_id]
    episode_command= ['--num-episode', str(num_eps)]
    policy_command = ['--log-custom-policy',teacher_path]
    if(episode==-1):
        eps_policy = []
    else:
        eps_policy = ['--episode', str(episode)]

    command=command_line + cpu_command +policy_command+name_command+env_command+task_command +episode_command +eps_policy
    if(task_id=='-sc' or task_id=='--simple-continual'):
        command+=['--short-episodes']
    ok=subprocess.call(command)



def allPolicy(log_dir):
    train_args, algo_name, algo_class, srl_model_path, env_kwargs = loadConfigAndSetup(log_dir)
    files= glob.glob(os.path.join(log_dir+algo_name+'_*_model.pkl'))
    files_list = []
    for file in files:
        eps=int((file.split('_')[-2]))
        files_list.append((eps,file))

    def sortFirst(val):
        return val[0]

    files_list.sort(key=sortFirst)
    res = np.array(files_list)
    return res[:,0], res[:,1]




def newPolicy(episodes, file_path):
    train_args, algo_name, algo_class, srl_model_path, env_kwargs=loadConfigAndSetup(file_path)
    episode, model_path,OK=latestPolicy(file_path,algo_name)
    if(episode in episodes):
        return -1,'', False
    else:
        return episode, model_path,True


def trainStudent(teacher_data_path,task_id,
                 yaml_file='config/srl_models.yaml',
                 log_dir='logs/',
                 srl_model='srl_combination',
                 env_name='OmnirobotEnv-v0',
                 training_size=40000, epochs=20):
    command_line = ['python','-m', 'rl_baselines.train','--latest','--algo', 'distillation','--log-dir',log_dir]
    srl_command  = ['--srl-model',srl_model]
    env_command  = ['--env',env_name]
    policy_command = ['--teacher-data-folder', teacher_data_path]
    size_epochs =['--distillation-training-set-size',str(training_size),'--epochs-distillation',str(epochs)]
    task_command = [task_id]
    ok = subprocess.call(command_line + srl_command
                         +env_command +policy_command +size_epochs+task_command +['--srl-config-file',yaml_file])

#python -m environments.dataset_fusioner
# --merge srl_zoo/data/circular_on_policy/ srl_zoo/data/reaching_on_policy/ srl_zoo/data/merge_CC_SC
def mergeData(teacher_dataset_1,teacher_dataset_2,merge_dataset):
    merge_command=['--merge',teacher_dataset_1,teacher_dataset_2,merge_dataset]
    subprocess.call(['python', '-m', 'environments.dataset_fusioner']+merge_command)



def evaluateStudent():
    #TODO

    return

if __name__ == '__main__':
    teacher_path='logs/circular/OmnirobotEnv-v0/srl_combination/ppo2/19-04-26_12h02_30/'
    teacher_path = 'logs_copy/cc2sc/OmnirobotEnv-v0/srl_combination/ppo2/19-04-26_20h17_37/'


    teacher_data_path='srl_zoo/data/circular_teacher/'
    task_id='-cc'
    output_name='circular_on_policy1'
    num_cpu=8
    yaml_file='config/srl_models_circular.yaml'
    merge_path='srl_zoo/data/merge'

    t1,t2='srl_zoo/data/Omnibot_circular','srl_zoo/data/Omnibot_random_simple'
    #mergeData(t1,t2,merge_path)
    #print(newPolicy([1,2,3],teacher_path))
    #trainStudent(teacher_data_path,task_id)


    task_pro,task_learn='-sc','-cc'
    teacher_pro   = 'logs_copy/cc2sc/OmnirobotEnv-v0/srl_combination/ppo2/19-04-26_20h17_37/'
    teacher_learn = 'logs_copy/cc2sc/OmnirobotEnv-v0/srl_combination/ppo2/19-04-26_20h17_37/'

    #The output path generate from the
    teacher_pro_data   = 'srl_zoo/data/Untitled Folder/'+task_pro[1:]+'/'
    teacher_learn_data = 'srl_zoo/data/Untitled Folder/'+task_learn[1:]+'/'



    episodes, policy_path = allPolicy(teacher_path)


    for eps in episodes:
        print('OK')
    #generate data from Professional teacher
    DatasetGenerator(teacher_pro,teacher_pro_data,task_pro,episodes[5],num_eps=200)
    #Generate data from learning teacher
    DatasetGenerator(teacher_learn, teacher_learn_data, task_learn, episodes[5],num_eps=200)
    #merge the data
    mergeData(teacher_pro_data, teacher_learn_data,merge_path)

    #train on the merged data
    log_dir = 'logs/student/'
    yaml_file ='config/srl_models.yaml'


    trainStudent(merge_path, task_id, yaml_file=yaml_file, log_dir=log_dir,
                 srl_model='srl_combination', env_name='OmnirobotEnv-v0',training_size=40000, epochs=20)

    evaluateStudent(log_dir)


