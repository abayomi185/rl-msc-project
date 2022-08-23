from typing import Optional

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from NiryoOne import NiryoOne

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.nucleus import get_server_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.torch import *

import numpy as np
import torch
import math

class NiryoOneTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._max_episode_length = 500
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        self._num_observations = 2
        self._num_actions = 5

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
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

        import omni.kit
        from pxr import UsdGeom
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.core.utils.stage import get_current_stage

        # from omni.isaac.core.utils.stage import print_stage_prim_paths
        # print_stage_prim_paths()

        self._sim_config.apply_articulation_settings("niryo_one", get_prim_at_path(niryo_one.prim_path), self._sim_config.parse_actor_config("niryo_one"))

        self._sim_config.apply_articulation_settings("visual_cube", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object"))

        super().set_up_scene(scene)

        self._niryo_ones = NiryoOneView(prim_paths_expr="/World/envs/.*/niryo_one", name="niryo_one_view")
        scene.add(self._niryo_ones)
        self._goals = RigidPrimView(prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view")
        scene.add(self._goals)

    def get_robot(self):
        return self._niryo_ones

    def get_observations(self):
        
        self._niryo_ones.

        observations = {
            self._niryo_ones.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        alpha = 2 * math.pi * np.random.rand()
        r = 0.25 * math.sqrt(np.random.rand()) + 0.1
        h = 0.6 * math.sqrt(np.random.rand())

        self._goals.set_world_poses(positions=np.array([math.sin(alpha) * r, math.cos(alpha) * r, h]), indices=env_ids)

        # self._niryo_ones.set_joint_positions()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self._robots = self.get_robot()
        self.goal_arm_position, _ = self._goals.get_world_pose()
        self.current_arm_position = self._niryo_ones.get_base_gripper_position()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._niryo_ones.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        self.rew_buf[:], self.reset_buf[:] = compute_task_reward(self.rew_buf, self.reset_buf, self.goal_pos, self.actions, self._max_episode_length)

        diff = 

        # reward = 

        self.rew_buf[:] = reward

    def is_done(self):
        pass





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




@torch.jit.script
def compute_task_reward(
    rew_buf, reset_buf, goal_pos, actions, max_episode_length
):
    # self.rew_buf, self.reset_buf, self.goal_pos, self.actions, self._max_episode_length


    return rewards, reset_buf