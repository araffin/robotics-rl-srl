import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from pytorch_agents.arguments import get_args
from pytorch_agents.envs import make_env
from pytorch_agents.kfac import KFACOptimizer
from pytorch_agents.model import CNNPolicy, MLPPolicy
from pytorch_agents.storage import RolloutStorage
import environments.kuka_button_gym_env as kuka_env
import rl_baselines.train as train
from rl_baselines.utils import filterJSONSerializableObjects

# To deal with using a second cameraargs = get_args()

train.LOG_INTERVAL = args.vis_interval
train.SAVE_INTERVAL = 100
train.ALGO = args.algo + "_pytorch"

args = train.configureEnvAndLogFolder(args, kuka_env)
train.saveEnvParams(kuka_env)

args_dict = filterJSONSerializableObjects(vars(args))
# Save args
with open(args.log_dir + "args.json", "w") as f:
    json.dump(args_dict, f)

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_cpu

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    if kuka_env.USE_SRL and not args.no_cuda:
        assert args.num_cpu == 1, "Multiprocessing not supported for srl models with CUDA (for pytorch_agents)"

    envs = [make_env(args.env, args.seed, i, args.log_dir)
            for i in range(args.num_cpu)]

    # Use multiprocessing only when needed
    # it prevents from a cuda error
    if args.num_cpu == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    obs_shape = envs.observation_space.shape


    # Check if we are using raw pixels or srl models

    if len(obs_shape) > 0:
        # In pytorch, images are channel first
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

    rollouts = RolloutStorage(args.num_steps, args.num_cpu, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_cpu, *obs_shape)

    def update_current_obs(observation):
        """
        Update the current observation:
        Convert numpy array to torch tensor and stack observations if needed
        :param observation: (numpy tensor)
        :return: (torch tensor)
        """
        n_channels = envs.observation_space.shape[0]
        obs_tensor = torch.from_numpy(observation).float()
        if args.num_stack > 1:
            current_obs[:, :-n_channels] = current_obs[:, n_channels:]
        current_obs[:, -n_channels:] = obs_tensor


    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            # Masks and states are only used for recurrent policies
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observe reward and next obs
            # dones is a list of bool (size = num processes)
            # rewards is tensor of size = num_processes x 1
            obs, rewards, dones, info = envs.step(cpu_actions)
            rewards = torch.from_numpy(np.expand_dims(np.stack(rewards), 1)).float()

            # If done then clean the history of observations.
            masks = torch.from_numpy(np.array(dones).astype(np.float32).reshape(-1, 1))

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            # Update the stacked observations
            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, rewards,
                            masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        # Updates for the different algorithms
        if args.algo in ['a2c', 'acktr']:
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                Variable(rollouts.masks[:-1].view(-1, 1)),
                Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_cpu, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_cpu, 1)

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

        train.callback(locals(), globals())


if __name__ == "__main__":
    main()
