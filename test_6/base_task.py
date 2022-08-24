from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim

import omni.kit

from omni.isaac.cloner import GridCloner
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.tasks.utils.usd_utils import create_distant_light
from omni.isaac.core.utils.nucleus import get_server_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.torch import *

from NiryoOne import NiryoOne

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

        # task-specific parameters
        self._robot_position = np.array([0.0, 0.0, 0.0])
        self._robot_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # values used for defining RL buffers
        self._num_observations = 1
        self._num_actions = 5
        self._device = "cuda"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 4), dtype=np.uint8)

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

    def set_up_scene(self, scene) -> None:

        scene.add_default_ground_plane()

        self._stage = get_current_stage()

        assets_root_path = get_server_path()

        niryo_asset_path = assets_root_path + "/niryo/mod_niryo_one_gripper1_n_camera/niryo_one_gripper1_n_camera.usd"

        niryo_one = NiryoOne(
            prim_path="/niryo_one",
            name="niryo_one",
            arm_dof_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
            usd_path=niryo_asset_path,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        goal = VisualCuboid(
            prim_path="/new_cube_1",
            name="visual_cube",
            position=np.array([0.60, 0.30, 0.025]),
            size=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0, 0]))

        self._niryo_ones = NiryoOneView(prim_paths_expr="/World/envs/.*/niryo_one", name="niryo_one_view")
        scene.add(self._niryo_ones)
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view")
        scene.add(self._goals)

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset(indices)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self):
        dof_pos = self._cartpoles.get_joint_positions()
        dof_vel = self._cartpoles.get_joint_velocities()

        # collect pole and cart joint positions and velocities for observation
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs[:, 0] = cart_pos
        self.obs[:, 1] = cart_vel
        self.obs[:, 2] = pole_pos
        self.obs[:, 3] = pole_vel

        return self.obs

    def calculate_metrics(self) -> None:
        cart_pos = self.obs[:, 0]
        cart_vel = self.obs[:, 1]
        pole_angle = self.obs[:, 2]
        pole_vel = self.obs[:, 3]

        # compute reward based on angle of pole and cart velocity
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # apply a penalty if cart is too far from center
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # apply a penalty if pole is too far from upright
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        return reward.item()

    def is_done(self) -> None:
        cart_pos = self.obs[:, 0]
        pole_pos = self.obs[:, 2]

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        self.resets = resets

        return resets.item()


class NiryoOneView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "ShadowHandView",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )