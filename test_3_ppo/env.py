from array import array
from typing import Optional, List, Dict

import gym
from gym import spaces
import numpy as np
import math
import carb

class NiryoOneEnv(gym.Env):
    # metadata: Dict[str, List[str]] = {"render.modes": ["human"]}
    metadata: Dict[str, List[str]] = {"render.modes": ["human"], "renderer": "RayTracedLighting"}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        from omni.isaac.core import World
        from NiryoOne import NiryoOne
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_server_path

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        # No need for Ground Plane as USD already has it
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_server_path()

        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        # niryo_asset_path = assets_root_path + "/niryo/niryo_one_main.usd"
        # niryo_asset_path = assets_root_path + "/niryo/niryo_one_mod.usd"
        niryo_asset_path = assets_root_path + "/niryo/niryo_one_gripper1_n_camera/niryo_one_gripper1_n_camera.usd"

        # TODO - Pass arm_dof_names
        self.niryo = self._my_world.scene.add(
            NiryoOne(
                prim_path="/niryo_one",
                name="my_niryo",
                arm_dof_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
                usd_path=niryo_asset_path,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.025]),
                size=np.array([0.08, 0.08, 0.08]),
                color=np.array([1.0, 0, 0]),
            )
        )

        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self.viewport_window_2 = None
        self._set_cameras()

        self.reward_range = (-float("inf"), float("inf"))
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

        # self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)

        self.observation_space = spaces.Box(low=0, high=255, shape=(256, 256, 4), dtype=np.uint8)

        # Two images in dict
        # self.observation_space = spaces.Dict(
        #     spaces={
        #         "cam_realsense": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
        #         "cam_end_effector": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
        #     }
        # )

        # Image and depth dict
        # self.observation_space = spaces.Dict(
        #     spaces={
        #         "realsense_vision": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
        #         "realsense_depth": spaces.Box(low=0, high=255, shape=(256, 256), dtype=np.uint8),
        #     }
        # )

        gym.Env.__init__(self)

        return

    def get_dt(self):
        return self._dt

    # TODO
    def step(self, action):
        # previous_arm_position, _ = self.niryo.get_world_pose()
        previous_arm_position = self.niryo.get_base_gripper_position()

        # print("+++Action: ")
        # print(action)

        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction

            self.niryo.apply_arm_actions(ArticulationAction(joint_positions=action * 10.0))
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True

        goal_arm_position, _ = self.goal.get_world_pose()
        # TODO - this pose likely needs to change.
        # current_arm_position, _ = self.niryo.get_world_pose()
        current_arm_position = self.niryo.get_base_gripper_position()

        # print("===========")
        # print("position")
        # print(goal_arm_position)
        # print(current_arm_position)
        # print("===========")

        previous_dist_to_goal = np.linalg.norm(goal_arm_position - previous_arm_position)

        current_dist_to_goal = np.linalg.norm(goal_arm_position - current_arm_position)

        test_reward = np.linalg.norm(goal_arm_position - current_arm_position)

        # print("+++++++++++")
        # print("diff")
        # print(previous_dist_to_goal)
        # print(current_dist_to_goal)
        # print("+++++++++++")

        # reward = (previous_dist_to_goal - current_dist_to_goal)

        addon_reward = (1/test_reward) if (1/test_reward) < 0.5 else -(1/test_reward)

        # reward = (previous_dist_to_goal - current_dist_to_goal) + addon_reward

        reward = (previous_dist_to_goal - current_dist_to_goal)

        # increase reward for being closer to the cube
        # reward = (previous_dist_to_goal - current_dist_to_goal) + (10/test_reward)

        # reward = 1 / (test_reward ** 2)

        # print("------------")
        # # print(reward)
        # print(test_reward)
        # print("------------")

        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 0.25 * math.sqrt(np.random.rand()) + 0.1
        h = 0.6 * math.sqrt(np.random.rand())
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, h]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB

        gt = self.sd_helper.get_groundtruth(
            ["rgb", "depth"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )

        # gt = self.sd_helper.get_groundtruth(
        #     ["rgb"], self.viewport_window_2, verify_sensor_init=False, wait_for_sensor_data=0
        # )

        # print(gt2["depth"].shape) # the width and height of the image

        # return gt["rgb"][:, :, :3], gt2["rgb"][:, :, :3]

        # return np.pad(np.concatenate((
        #     gt["rgb"][:, :, :3],
        #     gt2["rgb"][:, :, :3]), axis=0),
        #     pad_width=[(0, 0), (64, 64), (0, 0)],
        #     mode='constant')

        # return {
        #     "realsense_vision": gt["rgb"][:, :, :3],
        #     "realsense_depth": gt["depth"][:, :]
        # }

        gt_depth = np.multiply(gt["depth"][:, :], 100)

        # print(gt["depth"][:, :])

        return np.dstack((gt["rgb"][:, :, :3], gt_depth))


    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    # TODO - Kinda Done
    def _set_cameras(self):
        import omni.kit
        from pxr import UsdGeom
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.core.utils.stage import get_current_stage

        # from omni.isaac.core.utils.stage import print_stage_prim_paths
        # print_stage_prim_paths()

        camera_path_1 = "/niryo_one/base_link/realsense"
        camera_1 = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path_1))
        camera_1.GetClippingRangeAttr().Set((0.01, 10000))
        camera_1.GetHorizontalApertureAttr().Set(69.4)
        camera_1.GetVerticalApertureAttr().Set(42.5)
        camera_1.GetFocalLengthAttr().Set(60)
        camera_1.GetFocusDistanceAttr().Set(30)

        camera_path_2 = "/niryo_one/camera_link/end_effector_camera"
        camera_2 = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path_2))
        camera_2.GetClippingRangeAttr().Set((0.01, 10000))
        camera_2.GetHorizontalApertureAttr().Set(12.8)
        camera_2.GetVerticalApertureAttr().Set(11.6)
        camera_2.GetFocalLengthAttr().Set(10)
        camera_2.GetFocusDistanceAttr().Set(5.0)


        if self.headless:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path_1))
            viewport_window = viewport_handle.get_viewport_window()

            viewport_handle_2 = omni.kit.viewport_legacy.get_viewport_interface()
            # viewport_handle_2.get_viewport_window().set_active_camera(str(camera_path_2))
            # viewport_window_2 = viewport_handle_2.get_viewport_window()

            self.viewport_window = viewport_window
            # self.viewport_window_2 = viewport_window_2

            viewport_window.set_texture_resolution(256, 256)
            # viewport_window_2.set_texture_resolution(128, 128)

        # TODO - Implement changes for multi-camera for non-headless here
        else:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
            viewport_handle_2 = omni.kit.viewport_legacy.get_viewport_interface().get_instance('Viewport')

            viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window_2 = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle_2)

            viewport_window.set_active_camera(camera_path_1)
            # viewport_window_2.set_active_camera(camera_path_2)

            viewport_window.set_texture_resolution(256, 256)
            # viewport_window_2.set_texture_resolution(256, 256)

            viewport_window.set_window_pos(1000, 400)
            viewport_window.set_window_size(420, 420)
            # viewport_window_2.set_window_pos(800, 400)
            # viewport_window_2.set_window_size(420, 420)

            self.viewport_window = viewport_window
            # self.viewport_window_2 = viewport_window_2

        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb", "depth"], viewport=self.viewport_window)
        # self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window_2)
        self._my_world.render()
        self.sd_helper.get_groundtruth(["rgb", "depth"], self.viewport_window)
        # self.sd_helper.get_groundtruth(["rgb"], self.viewport_window_2)
        return
