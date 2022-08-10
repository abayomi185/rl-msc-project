from typing import Optional, List
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import ArticulationGripper
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
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

        # end_effector_prim_name: Optional[str] = None,
        # gripper_open_position: Optional[np.ndarray] = None,
        # gripper_closed_position: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                carb.log_error("No valid usd path defined to create Niryo One.")
        else:
            carb.log_error("no prim at path %s", prim_path)

        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        self._arm_dof_names = arm_dof_names
        self._arm_dof_indices = arm_dof_indices
        return

    @property
    def arm_dof_indices(self) -> np.ndarray:
        return self._arm_dof_indices

    def get_arm_position(self):
        full_dofs_positions = self.get_joint_positions()

    def set_arm_position(self, positions) -> None:
        pass

    def get_arm_velocity(self):
        pass

    def set_arm_velocity(self, velocities) -> None:
        pass

    def apply_arm_actions(self, actions: ArticulationAction) -> None:
        pass

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view=physics_sim_view)

        # TODO
        if self._arm_dof_names is not None:
            self._arm_dof_names = [
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

# Notes
# Add Articulation root to the base of the robot;
# See https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gui_simple_robot.html

