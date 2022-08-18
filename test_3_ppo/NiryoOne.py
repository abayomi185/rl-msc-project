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

    # def get_arm_velocity(self):
    #     pass

    # def set_arm_velocity(self, velocities) -> None:
    #     pass

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
