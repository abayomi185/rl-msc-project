from array import array
from typing import Optional, List, Dict

import gym
from gym import spaces
import numpy as np
import math
import carb

class NiryoOneEnv(gym.Env):
    metadata: Dict[str, List[str]] = {"render.modes": ["human"]}

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
        from test_2.NiryoOne import NiryoOne
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_server_path

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        # No need for Ground Plane as USD already has it
        # self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_server_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        niryo_asset_path = assets_root_path + "/niryo/niryo_one_main.usd"
        
        # TODO
        self.niryo = self._my_world.scene.add(
            NiryoOne(
                prim_path="/niryo",
                name="my_niryo",
                arm_dof_names=[],
                usd_path=niryo_asset_path,
                position=np.array([]),
                orientation=np.array([]),
            )
        )
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([0.60, 0.30, 0.025]),
                size=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0, 0]),
            )
        )
        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_camera()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        return

    def get_dt(self):
        return self._dt

    # TODO
    def step(self, action):
        previous_arm_position, _ = self.niryo.get_world_pose()
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
        current_arm_position, _ = self.niryo.get_world_pose()

        previous_dist_to_goal = np.linalg.norm(goal_arm_position - previous_arm_position)
        current_dist_to_goal = np.linalg.norm(goal_arm_position - current_arm_position)

        reward = previous_dist_to_goal - current_dist_to_goal
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 1.00 * math.sqrt(np.random.rand()) + 0.20
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.025]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return gt["rgb"][:, :, :3]

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    # TODO
    def _set_camera(self):
        import omni.kit
        from pxr import UsdGeom
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.core.utils.stage import get_current_stage

        camera_path = "/RealSense"
        camera = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path))
        camera.GetClippingRangeAttr().Set((0.01, 10000))

        if self.headless:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            self.viewport_window = viewport_window
            viewport_window.set_texture_resolution(128, 128)
        else:
            viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
            new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(
                viewport_handle
            )
            viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_texture_resolution(128, 128)
            viewport_window.set_window_pos(1000, 400)
            viewport_window.set_window_size(420, 420)
            self.viewport_window = viewport_window

        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        self._my_world.render()
        self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        return
