from typing import Optional, List, Dict

# import gym
# from omni.isaac.kit import SimulationApp
# import carb
import torch
import numpy as np

from omni.isaac.gym.vec_env import VecEnvBase


# env = VecEnvBase(headless=False)

class NiryoOneVecEnv(VecEnvBase):
    def step(self, actions):
        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()
        self._task.pre_physics_step(actions)

        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        # Details here changed to match example in SKRL as opposed to Isaac Gym

        observations, rewards, dones, info = self._task.post_physics_step()

        return {"obs": torch.clamp(observations, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()}, \
            rewards.to(self._task.rl_device).clone(), dones.to(self._task.rl_device).clone(), info.copy()

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict