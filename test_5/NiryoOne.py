from typing import Optional, List
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import ArticulationGripper
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
import carb

class NiryoOne(Robot):
    def __init__(
        self,
        prim_path: str,
        name: str = "niryo_one_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        # create_robot: Optional[bool] = False,
        arm_dof_names: Optional[List[str]] = None,
        arm_dof_indices: Optional[int] = None,

        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                # add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("No valid usd path defined to create Niryo One.")
        else:
            carb.log_error("no prim at path %s", prim_path)

        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        self._arm_dof_names = arm_dof_names
        self._arm_dof_indices = arm_dof_indices
        self.set_cameras()

        # Enables Gripper
        if gripper_dof_names is not None:
            self._gripper = ArticulationGripper(
                gripper_dof_names=gripper_dof_names,
                gripper_open_position=gripper_open_position,
                gripper_closed_position=gripper_closed_position
            )
        return

    @property
    def arm_dof_indices(self) -> np.ndarray:
        return self._arm_dof_indices

    def get_arm_positions(self):
        full_dofs_positions = self.get_joint_positions()
        arm_joint_positions = [full_dofs_positions[i] for i in self._arm_dof_indices]
        return arm_joint_positions

    def set_arm_position(self, positions) -> None:
        full_dofs_positions = [None] * self.num_dof
        for i in range(self._num_arm_dof):
            full_dofs_positions[self._arm_dof_indices[i]] = positions[i]
        self.set_joint_positions(positions=np.array(full_dofs_positions))
        return

    def get_base_gripper_position(self) -> np.ndarray:
        object = self._dc_interface.get_rigid_body("/niryo_one/base_gripper_1")
        object_pose = self._dc_interface.get_rigid_body_pose(object)
        return object_pose.p

    def apply_arm_actions(self, actions: ArticulationAction) -> None:
        actions_length = actions.get_length()

        # print("++++++++++")
        # print(actions)
        # print(self._num_arm_dof)
        # print("++++++++++")

        if actions_length is not None and actions_length != self._num_arm_dof:
            raise Exception("ArticulationAction passed should be the same length as the number of wheels")

        joint_actions = ArticulationAction()
        if actions.joint_positions is not None:
            joint_actions.joint_positions = np.zeros(self.num_dof)  # for all dofs of the robot
            for i in range(self._num_arm_dof):  # set only the ones that are the wheels
                joint_actions.joint_positions[self._arm_dof_indices[i]] = actions.joint_positions[i]

        self.apply_action(control_actions=joint_actions)
        return

    def initialize(self, physics_sim_view=None) -> None:
        # print("++++++++++")
        # print(self.prim_path)
        super().initialize(physics_sim_view=physics_sim_view)
        # print("++++++++++")

        # print(self._arm_dof_names)

        # TODO
        if self._arm_dof_names is not None:
            self._arm_dof_indices = [
                self.get_dof_index(self._arm_dof_names[i]) for i in range(len(self._arm_dof_names))
            ]
        elif self._arm_dof_indices is None:
            carb.log_error("need to have either joint names or joint indices")

        self._num_arm_dof = len(self._arm_dof_indices)

        return

    def post_reset(self) -> None:
        super().post_reset()
        self._articulation_controller.switch_control_mode(mode="position")
        # self._articulation_controller.switch_dof_control_mode(dof_index=self.gripper.dof_indices[0], mode="position")
        # self._articulation_controller.switch_dof_control_mode(dof_index=self.gripper.dof_indices[1], mode="position")
        return

    def get_articulation_controller_properties(self):
        return self._arm_dof_names, self._arm_dof_indices

    def set_cameras(self):
        import omni.kit
        from pxr import UsdGeom
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        from omni.isaac.core.utils.stage import get_current_stage

        # from omni.isaac.core.utils.stage import print_stage_prim_paths
        # print_stage_prim_paths()

        camera_path_1 = self.prim_path + "/base_link/realsense"
        camera_1 = UsdGeom.Camera(get_current_stage().GetPrimAtPath(camera_path_1))
        camera_1.GetClippingRangeAttr().Set((0.01, 10000))
        camera_1.GetHorizontalApertureAttr().Set(69.4)
        camera_1.GetVerticalApertureAttr().Set(42.5)
        camera_1.GetFocalLengthAttr().Set(50)
        camera_1.GetFocusDistanceAttr().Set(30)

        camera_path_2 = self.prim_path + "/camera_link/end_effector_camera"
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

            # viewport_window.set_window_pos(1000, 400)
            # viewport_window.set_window_size(420, 420)
            # viewport_window_2.set_window_pos(800, 400)
            # viewport_window_2.set_window_size(420, 420)

            self.viewport_window = viewport_window
            # self.viewport_window_2 = viewport_window_2

        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb", "depth"], viewport=self.viewport_window)
        # self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window_2)
        # self._my_world.render()
        # self.sd_helper.get_groundtruth(["rgb", "depth"], self.viewport_window)
        # self.sd_helper.get_groundtruth(["rgb"], self.viewport_window_2)
        return
