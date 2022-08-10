# For type annotations
from __future__ import annotations
from typing import Optional

# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import math
import torch

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb

class NiryoOne(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "NiryoOne",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find assets root path")
                self._usd_path = assets_root_path + "/niryo/niryo_one.usd"
                # omniverse://127.0.0.1/niryo/

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(prim_path, name, translation, orientation, articulation_controller=None)
