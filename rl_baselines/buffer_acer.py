import numpy as np


# Modified version of OpenAI Baselines ACER buffer to work with srl models
class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, env, nsteps, nstack, size=50000):
        self.nenv = env.num_envs
        self.nsteps = nsteps
        self.nstack = nstack
        self.nbatch = self.nenv * self.nsteps
        # Each loc contains nenv * nsteps frames, thus total buffer is nenv * size frames
        self.size = size // self.nsteps

        if len(env.observation_space.shape) > 1:
            self.raw_pixels = True
            self.nh, self.nw, self.nc = env.observation_space.shape
            self.obs_dtype = np.uint8
        else:
            self.raw_pixels = False
            self.obs_dim = env.observation_space.shape[0]
            self.obs_dtype = np.float32

        # Memory
        self.enc_obs = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None
        self.masks = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.num_in_buffer >= (frames // self.nsteps)

    def can_sample(self):
        return self.num_in_buffer > 0

    # Generate stacked frames
    def decode(self, enc_obs, dones):
        # enc_obs has shape [nenvs, nsteps + nstack, nh, nw, nc]
        # dones has shape [nenvs, nsteps, nh, nw, nc]
        # returns stacked obs of shape [nenv, (nsteps + 1), nh, nw, nstack*nc]
        if self.raw_pixels:
            nstack, nenv, nsteps, nh, nw, nc = self.nstack, self.nenv, self.nsteps, self.nh, self.nw, self.nc
            x = np.reshape(enc_obs, [nstack, nenv, nsteps + nstack, nh, nw, nc]).swapaxes(2, 1)
            return np.reshape(x[:, :].transpose((2, 1, 3, 4, 0, 5)), [nenv, (nsteps + 1), nh, nw, nstack * nc])
        else:
            nstack, nenv, nsteps, obs_dim = self.nstack, self.nenv, self.nsteps, self.obs_dim
            x = np.reshape(enc_obs, [nstack, nenv, nsteps + nstack, obs_dim]).swapaxes(2, 1)
            return np.reshape(x[:, :].transpose((2, 1, 3, 0)), [nenv, (nsteps + 1), obs_dim, nstack])

    def put(self, enc_obs, actions, rewards, mus, dones, masks):
        # enc_obs [nenv, (nsteps + nstack), nh, nw, nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]

        if self.enc_obs is None:
            self.enc_obs = np.empty([self.size] + list(enc_obs.shape), dtype=self.obs_dtype)
            self.actions = np.empty([self.size] + list(actions.shape), dtype=np.int32)
            self.rewards = np.empty([self.size] + list(rewards.shape), dtype=np.float32)
            self.mus = np.empty([self.size] + list(mus.shape), dtype=np.float32)
            self.dones = np.empty([self.size] + list(dones.shape), dtype=np.bool)
            self.masks = np.empty([self.size] + list(masks.shape), dtype=np.bool)

        self.enc_obs[self.next_idx] = enc_obs
        self.actions[self.next_idx] = actions
        self.rewards[self.next_idx] = rewards
        self.mus[self.next_idx] = mus
        self.dones[self.next_idx] = dones
        self.masks[self.next_idx] = masks

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, x, idx, envx):
        nenv = self.nenv
        out = np.empty([nenv] + list(x.shape[2:]), dtype=x.dtype)
        for i in range(nenv):
            out[i] = x[idx[i], envx[i]]
        return out

    def get(self):
        # returns
        # obs [nenv, (nsteps + 1), nh, nw, nstack*nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]
        nenv = self.nenv
        assert self.can_sample()

        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, nenv)
        envx = np.arange(nenv)

        take = lambda x: self.take(x, idx, envx)  # for i in range(nenv)], axis = 0)
        dones = take(self.dones)
        enc_obs = take(self.enc_obs)
        obs = self.decode(enc_obs, dones)
        actions = take(self.actions)
        rewards = take(self.rewards)
        mus = take(self.mus)
        masks = take(self.masks)
        return obs, actions, rewards, mus, dones, masks
