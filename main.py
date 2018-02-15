import copy
import glob
import os
import time
import sys
from datetime import datetime

import gym
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from pytorch_agents.arguments import get_args
from pytorch_agents.envs import make_env
from pytorch_agents.kfac import KFACOptimizer
from pytorch_agents.model import CNNPolicy, MLPPolicy
from pytorch_agents.storage import RolloutStorage
from pytorch_agents.visualize import visdom_plot, episode_plot
import environments
import environments.kuka_button_gym_env as kuka_env

kuka_env.ACTION_REPEAT = 4

args = get_args()

with open('config/srl_models.yaml', 'rb') as f:
    models = yaml.load(f)

PLOT_TITLE = "Raw Pixels"

if args.srl_model != "":
    PLOT_TITLE = args.srl_model
    path = models.get(args.srl_model)
    args.log_dir += args.srl_model + "/"

    if args.srl_model == "ground_truth":
        kuka_env.USE_GROUND_TRUTH = True
        PLOT_TITLE = "Ground Truth"
    elif path is not None:
        kuka_env.USE_SRL = True
        kuka_env.SRL_MODEL_PATH = models['log_folder'] + path
    else:
        raise ValueError("Unsupported value for srl-model: {}".format(args.srl_model))

else:
    args.log_dir += "raw_pixels/"

# Add date + current time
args.log_dir += datetime.now().strftime("%d-%m-%y_%Hh%M")

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

os.makedirs(args.log_dir, exist_ok=True)


def main():
    print("#######")
    print("""WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py)
           or visdom plot to get true rewards""")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win, win_smooth = None, None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    print(obs_shape)

    if len(obs_shape) > 0:
        obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    else:
        obs_shape = (args.num_stack, *obs_shape[0])

    if len(envs.observation_space.shape) == 3:
        print("Using CNNPolicy")
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy, input_dim=obs_shape[1])
    else:
        print("Using MLPPolicy")
        assert not args.recurrent_policy, \
            "Recurrent policy is not implemented for the MLP controller"
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next obs
            # done is a list of bool (size = num processes)
            # reward is tensor of size = num_processes x 1
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward,
                            masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                Variable(rollouts.masks[:-1].view(-1, 1)),
                Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages,
                                                                  args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages,
                                                                     args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                        Variable(observations_batch),
                        Variable(states_batch),
                        Variable(masks_batch),
                        Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and  args.log_dir != "":
            save_path = os.path.join(args.log_dir, args.algo)
            os.makedirs(save_path, exist_ok=True)

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pth"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            if args.vis:
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo, bin_size=1, smooth=0, title=PLOT_TITLE)

            print(
                "Updates {}, num timesteps {}, FPS {} entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)), dist_entropy.data[0], value_loss.data[0], action_loss.data[0]))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win_smooth = visdom_plot(viz, win_smooth, args.log_dir, args.env_name, args.algo, title=PLOT_TITLE + " smoothed")
            except IOError as e:
                pass


if __name__ == "__main__":
    main()
