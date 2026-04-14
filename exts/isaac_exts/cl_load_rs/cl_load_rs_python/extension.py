# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import gc
import omni
import omni.usd

from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE

from .realsense import Spec, CamType, CameraObjectFactory, ROS2CameraFactory 

"""
This file serves as a basic template for the standard boilerplate operations
that make a UI-based extension appear on the toolbar.

This implementation is meant to cover most use-cases without modification.
Various callbacks are hooked up to a seperate class UIBuilder in .ui_builder.py
Most users will be able to make their desired UI extension by interacting solely with
UIBuilder.

This class sets up standard useful callback functions in UIBuilder:
    on_menu_callback: Called when extension is opened
    on_timeline_event: Called when timeline is stopped, paused, or played
    on_physics_step: Called on every physics step
    on_stage_event: Called when stage is opened or closed
    cleanup: Called when resources such as physics subscriptions should be cleaned up
    build_ui: User function that creates the UI they want.
"""


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""
        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()

        # Events
        #self._usd_context = omni.usd.get_context()
        #self._physxIFace = _physx.get_physx_interface()
        #self._physx_subscription = None
        #self._stage_event_sub = None
        #self._timeline = omni.timeline.get_timeline_interface()

        #publish realsense rgb and depth data
        l_depth = Spec(
                name = "l_depth",
                path = "/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/L_hand_base_link/CL_L_realsense/rsd455/RSD455/Camera_Pseudo_Depth",
		role = CamType.DEPTH
                )
        l_rgb = Spec(
                name = "l_rgb",
                path = "/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/L_hand_base_link/CL_L_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
		role = CamType.RGB
                )
        r_depth = Spec(
                name = "r_depth",
                path = "/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_Pseudo_Depth",
		role = CamType.DEPTH
                )
        r_rgb = Spec(
                name = "r_rgb",
                path = "/World/envs/env_0/Robot/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense/R_hand_base_link/CL_R_realsense/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
		role = CamType.RGB
                )
        specs = [l_depth, l_rgb, r_depth, r_rgb]
        pkg_specs, camera_objects = CameraObjectFactory(specs).export()
        print(f"{camera_objects=}")
        print(f"{pkg_specs=}")
        ros2_cameras = ROS2CameraFactory(camera_objects, pkg_specs)
	

    def on_shutdown(self):
        gc.collect()

