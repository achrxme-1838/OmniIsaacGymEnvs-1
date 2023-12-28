# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.Z1Effort import Z1Effort
from omniisaacgymenvs.robots.articulations.views.z1_view import Z1View

from omni.isaac.core.objects import DynamicSphere

from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom

import time


class Z1Task(RLTask):
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

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        
        self.dt = 1/120.

        self._num_observations = 31
        self._num_actions = 7
        
        self._gripper_too_low_height = 0.11
        RLTask.__init__(self, name, env)
        
        self._z1_rand_scale = 0.3
        self._ball_base_position = torch.tensor([0.0, 0.0, 0.5]).to(self._device)
        
        # self._ball_radius = 0.02
        self._ball_radius = 0.01
        self._finger_vector = torch.tensor([0.09, 0.0, -0.02], device='cuda:0')
        
        # self.Kp = torch.tensor([11.4592, 40.4441, 20.8348, 19.7572, 22.3230, 10.7430, 38.1972], device='cuda:0')
        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()

        self.get_z1()
        self.add_ball()
        
        super().set_up_scene(scene)
        
        self._z1s = Z1View(prim_paths_expr="/World/envs/.*/z1", name="z1_view")
        scene.add(self._z1s)
        scene.add(self._z1s._link00)
        scene.add(self._z1s._link01)
        scene.add(self._z1s._link02)
        scene.add(self._z1s._link03)
        scene.add(self._z1s._link04)
        scene.add(self._z1s._link05)
        scene.add(self._z1s._gripperStator)
        scene.add(self._z1s._gripperMover)
    
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/Ball/ball", name="ball_view", reset_xform_properties=False)
        scene.add(self._balls)  
        
        self.init_data()
        
        return

    def get_z1(self):
        self._z1 = Z1Effort(prim_path=self.default_zero_env_path + "/z1", name="z1")
        self._sim_config.apply_articulation_settings("z1", get_prim_at_path(self._z1.prim_path), self._sim_config.parse_actor_config("z1"))

        self._z1.set_z1_properties(self._stage, self._z1.prim)
        
    def add_ball(self):
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/Ball/ball", 
            translation=self._ball_base_position, 
            name="ball_0",
            radius=self._ball_radius,
            color=torch.tensor([1.0, 0.0, 0.0]),
        )

        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        
    
    def init_data(self) -> None:

        self.z1_dof_pos = torch.zeros((self.num_envs, self._num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.z1_dof_vel = torch.zeros((self.num_envs, self._num_actions), dtype=torch.float, device=self.device, requires_grad=False)

        self.current_actions = torch.zeros((self.num_envs, self._num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros((self.num_envs, self._num_actions), dtype=torch.float, device=self.device, requires_grad=False)

        # self.z1_default_dof_pos = torch.tensor([0.0, 1.5, -1.5, 0.0, 0.0, 0.0, -0.8], dtype=torch.float, device=self.device, requires_grad=False)
        self.z1_default_dof_pos = torch.tensor([0.0, 1.0, -1.4, 0.0, 0.0, 0.0, -0.0], dtype=torch.float, device=self.device, requires_grad=False)
        
        self.gripper_mover_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_mover_rot = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.pinch_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        


    def get_observations(self) -> dict:

        z1_dof_pos = self._z1s.get_joint_positions(clone=False)
        z1_dof_vel = self._z1s.get_joint_velocities(clone=False)

        self.z1_dof_pos = z1_dof_pos
        self.z1_dof_vel = z1_dof_vel


        dof_pos_scaled = (
            2.0 * (z1_dof_pos - self.z1_dof_lower_limits)
            / (self.z1_dof_upper_limits - self.z1_dof_lower_limits)
            - 1.0
        )

        z1_dof_vel_scaled = z1_dof_vel * self.dof_vel_scale
        
        ball_pos, _ = self._balls.get_world_poses(clone=False)
        ball_pos = ball_pos - self._env_pos
        ball_pos_tensor = torch.tensor(ball_pos, dtype=torch.float32)
        
        gripper_mover_pos, gripper_mover_rot = self._z1s._gripperMover.get_world_poses(clone=False)
        gripper_mover_pos = gripper_mover_pos - self._env_pos
        self.gripper_mover_pos = torch.tensor(gripper_mover_pos, dtype=torch.float32)
        self.gripper_mover_rot = torch.tensor(gripper_mover_rot, dtype=torch.float32)
        
        gripper_mover_rot_mat = quats_to_rot_matrices(self.gripper_mover_rot)
        finger_offset = torch.matmul(gripper_mover_rot_mat, self._finger_vector)
        
        self.pinch_pos = self.gripper_mover_pos[:, :] + finger_offset
        
        self.obs_buf = torch.cat(
            (              
                self.current_actions, # 7
                
                dof_pos_scaled, # 7
                z1_dof_vel_scaled, # 7
                
                # self.gripper_mover_pos, # 3
                self.pinch_pos,  # 3
                self.gripper_mover_rot, # 4
                
                ball_pos_tensor, # 3
            ),
            dim=-1  
        )
        
        observations = {
            self._z1s.name: {
                "obs_buf": self.obs_buf
            }
        }
        
        return observations
    
    def refresh_dof_state_tensors(self):
        self.z1_dof_pos = self._z1s.get_joint_positions(clone=False)
        self.z1_dof_vel = self._z1s.get_joint_velocities(clone=False)
    
    def pre_physics_step(self, actions) -> None:
        
        if not self._env._world.is_playing():
            return

        # Reset is occured here
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.max_force = torch.tensor([30.0, 60.0, 30.0, 30.0, 30.0, 30.0, 30.0], device='cuda:0')
        self.dof_range = self.z1_dof_upper_limits - self.z1_dof_lower_limits
        
        self.Kp = self.max_force * 2.0 / self.dof_range
        self.Kp_scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device='cuda:0') # downsize the gripper gain
        self.Kp = self.Kp * self.Kp_scale
        self.Kd = torch.tensor([0.05] * 7, device='cuda:0')
        
        self.prev_actions = self.current_actions
        self.current_actions = actions.clone().to(self.device)
        
        targets = self.z1_dof_targets + self.current_actions
        self.z1_dof_targets[:] = tensor_clamp(targets, self.z1_dof_lower_limits, self.z1_dof_upper_limits)
        
        for i in range(2):
            if self._env._world.is_playing():       
                torques = self.Kp*(self.z1_dof_targets - self.z1_dof_pos) - self.Kd*self.z1_dof_vel
                self.controls = tensor_clamp(torques, -self.max_force, self.max_force)
                self._z1s.set_joint_efforts(self.controls)
                
                SimulationContext.step(self._env._world, render=False)
                self.refresh_dof_state_tensors()
        
        # to make the ball to gripper        
        # ball_positions, _ = self._balls.get_world_poses(clone=False)
        
        # gripper_mover_rot_mat = quats_to_rot_matrices(self.gripper_mover_rot)
        # finger_offset = torch.matmul(gripper_mover_rot_mat, self._finger_vector)
        
        # ball_positions[:, :] = self.gripper_mover_pos[:, :] + self._env_pos + finger_offset
        # self._balls.set_world_poses(positions=ball_positions)
        
    def generate_random_points(self, n_points):
        outer_box_size = 0.8
        inner_box_size = 0.2
        
        points = []
        while len(points) < n_points:
            point = outer_box_size * (torch.rand(2, device=self._device) - 0.5)
            if torch.abs(point[0]) > inner_box_size or torch.abs(point[1]) > inner_box_size:
                points.append(point)

        return torch.stack(points)


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)
        
        # RESET Robot
        # randomize based on default pose
        pos = tensor_clamp(
            self.z1_default_dof_pos.unsqueeze(0)
            + self._z1_rand_scale * (2 * torch.rand((len(env_ids), self.num_z1_dofs), device=self._device) - 1),
            self.z1_dof_lower_limits,   # rad
            self.z1_dof_upper_limits,
        )

        dof_pos = torch.zeros((num_indices, self._z1s.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._z1s.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.z1_dof_targets[env_ids, :] = pos
        self.z1_dof_pos[env_ids, :] = pos
        
        self._z1s.set_joint_position_targets(self.z1_dof_targets[env_ids], indices=indices)
        self._z1s.set_joint_positions(dof_pos, indices=indices)
        self._z1s.set_joint_velocities(dof_vel, indices=indices)
        
        # RESET Ball
        height_range = 0.4
        
        new_ball_pos = self.initial_ball_pos[env_ids]
        random_positions = self.generate_random_points(num_indices)

        new_ball_pos[:, 0] += random_positions[:, 0]
        new_ball_pos[:, 1] += random_positions[:, 1]

        random_z = torch.rand(num_indices, device=self._device)
        new_ball_pos[:, 2] += height_range * random_z - 0.5 * height_range
        
        ball_pos = torch.zeros((num_indices, 3), device=self._device)
        ball_pos[:, :] = new_ball_pos
        self._balls.set_world_poses(positions=ball_pos, indices=indices)
        
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    # Called only at the first of simulation
    def post_reset(self):
        self.num_z1_dofs = self._z1s.num_dof
        self.z1_dof_pos = torch.zeros((self.num_envs, self.num_z1_dofs), device=self._device)
        dof_limits = self._z1s.get_dof_limits()
        
        self.z1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.z1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        
        self.z1_dof_speed_scales = torch.ones_like(self.z1_dof_lower_limits)
        self.z1_dof_targets = torch.zeros(
            (self._num_envs, self.num_z1_dofs), dtype=torch.float, device=self._device
        )
        
        # self.initial_cube_pos, self.initial_cube_rot = self._balls.get_world_poses()
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()
        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)


    def calculate_metrics(self) -> None:

        self.rew_buf[:] = self.compute_z1_reward()

    def is_done(self) -> None:
        
        gripper_mover_pos, gripper_mover_rot = self._z1s._gripperMover.get_world_poses(clone=False)
        gripper_mover_pos = gripper_mover_pos - self._env_pos
        
        low_offset = 0.003
        
        gripper_too_low_condition = (gripper_mover_pos[:, 2] < self._gripper_too_low_height - low_offset)
        self.reset_buf = torch.where(gripper_too_low_condition, torch.ones_like(self.reset_buf), self.reset_buf)         # on the ground
        
        
        # reset when the episode reach max
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, 
                                     torch.ones_like(self.reset_buf), self.reset_buf)
    
    def compute_z1_reward(self):

        ball_positions = self._balls.get_world_poses(clone=False)[0].to(self._device)
        ball_positions = ball_positions - self._env_pos
        
        # gripper_ball_distance = torch.sqrt(torch.sum((self.gripper_mover_pos - ball_positions) ** 2, dim=1))
        pinch_ball_distance = torch.sqrt(torch.sum((self.pinch_pos - ball_positions) ** 2, dim=1))
    
        # print(pinch_ball_distance)
        
        rewards = 1.0 / (0.01 + pinch_ball_distance)
        
        # jitter penalize
        jitter_penal_scale = 1.0
        action_diff = self.current_actions - self.prev_actions
        # abs_action_diff = abs(action_diff)
        sum_abs_action_diff = torch.sum(abs(action_diff), dim=1)
        
        # print(sum_abs_action_diff)
        rewards = rewards - jitter_penal_scale * sum_abs_action_diff
        
        gripper_too_low_condition = (self.gripper_mover_pos[:, 2] < self._gripper_too_low_height).unsqueeze(0)
        rewards = torch.where(gripper_too_low_condition, torch.ones_like(rewards) * -10.0, rewards)
         
        
        # print("[Pos] : ", gripper_mover_pos)
        # print("[Rewards] : ", rewards)
        
        return rewards