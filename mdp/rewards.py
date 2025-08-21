# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.env import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.
    
    The function compes the position error between the desired postion (from the comand) and the
    current postion of the assets's body (in world fram). The psotion error is computed as the L2-norm
    of the difference between the desired and the crrent postions."""

    #extract the assets (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current postions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:,:3],asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(env:ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of the postion using the tanh kernel
    
    The function computes the postion error between the desired postion (from the command) and the 
    current postion of the asset's body (in the world frame) and maps it with a tanh kernel."""

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    #obtain the desired and current postions
    des_pos_b = command[:,:3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0],:3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    tanh = torch.tanh(distance/std)
    tanh = torch.where(tanh<0.1,-tanh,tanh)
    return 1 - torch.tanh(distance/std)

def orientation_error(env:ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of the postion using the tanh kernel
    
    The function computes the postion error between the desired postion (from the command) and the 
    current postion of the asset's body (in the world frame) and maps it with a tanh kernel."""

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    #obtain the desired and current postions
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0],3:7]
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    error = torch.where(error<0.5,error-1,error)
    return error