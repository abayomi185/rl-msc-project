# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})


from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim

import omni.kit

import gym
from gym import spaces
import numpy as np
import torch
import math

class NiryoOneTask(BaseTask):
    def __init__(
        self,
        name,
        offset=None
    ) -> None:
        self._niryo_position = [0.0, 0.0, 0.0]

        # values used for defining RL buffers
        self._num_observations = 4
        self._num_actions = 1
        self._device = "gpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        # Trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Niryo_One USD file
        assets_root_path = get_assets_root_path()
        print("\n\n" + assets_root_path + "\n\n")
        usd_path = assets_root_path + "/niryo/niryo_one.usd"

        # add the Niryo_One USD to our stage
        create_prim(prim_path="/World/Niryo", prim_type="Xform", position=self._niryo_position)
        add_reference_to_stage(usd_path, "/World/Niryo")

        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._niryo_robots = ArticulationView(prim_paths_expr="/World/Niryo*", name="niryo_view")

        # add Cartpole ArticulationView and ground plane to the Scene
        scene.add(self._niryo_robots)
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._niryo_robots.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._niryo_robots.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._niryo_robots.set_joint_positions(dof_pos, indices=indices)
        self._niryo_robots.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0
