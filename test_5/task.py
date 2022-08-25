from typing import Optional

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from NiryoOne import NiryoOne

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.core.utils.nucleus import get_server_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.torch import torch_rand_float, scale

from pxr import Gf, Usd, UsdGeom, UsdShade

import numpy as np
import torch
import math

import omni
import carb

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
        print(self._cfg)
        self._task_cfg = sim_config.task_config
        print(self._task_cfg)

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._max_episode_length = 500

        self._num_observations = 2
        self._num_actions = 5

        RLTask.__init__(self, name, env)

        # self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self._device)

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
        
        self.goal_pos, _ = self._goals.get_world_poses(clone=False)
        self.robot_pos = self._niryo_ones.get_base_gripper_positions()

        gts = self._niryo_ones.get_camera_ground_truths()
        self.obs_buf[:, 0:-1] = gts[0]

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

        self.actions = actions.clone().to(self._device)

        self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions, -10, 10)

        self._niryo_ones.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        def new_position():
            alpha = 2 * math.pi * np.random.rand()
            r = 0.25 * math.sqrt(np.random.rand()) + 0.1
            h = 0.6 * math.sqrt(np.random.rand())

            return (alpha, r, h)

        # rand_floats_1 = torch_rand_float(-6.3, 6.3, (num_resets, 1), device=self.device)
        # rand_floats_2 = torch_rand_float(-1.35, 1.35, (num_resets, 1), device=self.device)
        # rand_floats_3 = torch_rand_float(-0.6, 0.6, (num_resets, 1), device=self.device)

        new_goal_pos = np.array([[math.sin(alpha) * r, math.cos(alpha) * r, h] for (alpha, r, h) in new_position()])

        self._goals.set_world_poses(positions=new_goal_pos, indices=env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


    def post_reset(self):
        self.num_hand_dofs = self._niryo_ones.num_dof
        self.actuated_dof_indices = self._niryo_ones.actuated_dof_indices 
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self._device)

        # self._robots = self.get_robot()
        # self.goal_arm_position, _ = self._goals.get_world_pose()
        # self.current_arm_position = self._niryo_ones.get_base_gripper_position()

        self.goal_pos, _ = self._goals.get_world_poses().clone() - self._env_pos
        self.robot_pos = self._niryo_ones.get_base_gripper_positions()

        # self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._niryo_ones.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        self.rew_buf[:] = compute_task_reward(self.rew_buf, self.reset_buf, self.goal_pos, self.robot_pos, self.actions)

        # diff = 

        # reward = 

        # self.rew_buf[:] = reward

    def is_done(self):
        pass


@torch.jit.script
def compute_task_reward(
    rew_buf, reset_buf, goal_pos, robot_pos, actions
):
    # self.rew_buf, self.reset_buf, self.goal_pos, self.actions, self._max_episode_length

    goal_dist = torch.norm(robot_pos - goal_pos, p=2, dim=-1)
    
    dist_rew = goal_dist

    reward = dist_rew

    # return rewards, reset_buf
    return reward, 



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

        self.sd_helper = SyntheticDataHelper()

        self._base_grippers = RigidPrimView(prim_paths_expr="/World/envs/.*/niryo_one/base_gripper_1", name="finger_view")

    def get_camera_ground_truths(self, indices=None):
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        camera_groundtruths = self._backend_utils.create_zeros_tensor([indices.shape[0], 256, 256, 4], dtype="float32", device=self._device)        

        write_idx = 0
        for i in indices:

            camera_path = "/World/envs/." + str(i) + "/niryo_one/base_link/realsense"
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            
            gt = self.sd_helper.get_groundtruth(
                ["rgb", "depth"], 
                viewport_window,
                verify_sensor_init=False, wait_for_sensor_data=0
                )
            gt_depth = np.multiply(gt["depth"][:, :], 100)

            camera_groundtruths[write_idx] = np.dstack((gt["rgb"][:, :, :3], gt_depth))

            write_idx += 1
        return camera_groundtruths

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices

    @property
    def get_base_gripper_positions(self):
        self._base_gripper_positions, _ = self._base_grippers.get_world_poses()
        return self._base_gripper_positions

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self.actuated_joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self._actuated_dof_indices = list()
        for joint_name in self.actuated_joint_names:
            self._actuated_dof_indices.append(self.get_dof_index(joint_name))
        self._actuated_dof_indices.sort()
