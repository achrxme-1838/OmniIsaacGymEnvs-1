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

from tensorboardX import SummaryWriter

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

        self._num_observations = 25
        self._num_actions = 7
        
        self._gripper_too_low_height = 0.11
        RLTask.__init__(self, name, env)
        
        # self._z1_rand_scale = 1.0
        self._z1_rand_scale = 0.3
        
        self._ball_base_position = torch.tensor([0.0, 0.0, 0.5]).to(self._device)
        
        # for the random
        self._ball_radius = 0.01
        self.height_range = 0.3
        self.outer_box_size = 0.8
        self.inner_box_size = 0.3
        
        self._mover_finger_vector = torch.tensor([0.09, 0.0, -0.02], device='cuda:0')
        self._stator_finger_vector = torch.tensor([0.135, 0.0, -0.01], device='cuda:0')
        
        # self.Kp = torch.tensor([11.4592, 40.4441, 20.8348, 19.7572, 22.3230, 10.7430, 38.1972], device='cuda:0')
        
        self._success_threshold = self._ball_radius * 2.0         # 0.01 * 2 = 2 cm
        # self._success_threshold = self._ball_radius * 5.0         # 0.01 * 5 = 5 cm
        
        self._max_success_num = 1000
        self._success_idices = torch.zeros(self._num_envs).to(self._device)
        self.success_list = []
        self.max_success_rate = 0.0
        
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

        # gripper : -0.8 is half
        self.z1_default_dof_pos = torch.tensor([0.0, 1.0, -1.4, 0.0, 0.0, 0.0, -0.5], dtype=torch.float, device=self.device, requires_grad=False)
        
        self.gripper_mover_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_mover_rot = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_stator_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.gripper_stator_rot = torch.ones((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.mover_pinch_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.stator_pinch_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        

    def get_observations(self) -> dict:

        self.z1_dof_pos = self._z1s.get_joint_positions(clone=False)
        self.z1_dof_vel = self._z1s.get_joint_velocities(clone=False)
        
        scaled_dof_pos = (
            2.0 * (self.z1_dof_pos - self.z1_dof_lower_limits)
            / (self.z1_dof_upper_limits - self.z1_dof_lower_limits)
            - 1.0
        )
        
        scaled_dof_vel = self.dof_vel_scale * self.z1_dof_vel
        

        ball_pos, _ = self._balls.get_world_poses(clone=False)
        self.ball_pos = torch.tensor(ball_pos - self._env_pos)
        
        gripper_mover_pos, gripper_mover_rot = self._z1s._gripperMover.get_world_poses(clone=False)
        gripper_mover_pos = gripper_mover_pos - self._env_pos
        self.gripper_mover_pos = torch.tensor(gripper_mover_pos, dtype=torch.float32)
        self.gripper_mover_rot = torch.tensor(gripper_mover_rot, dtype=torch.float32)
        
        gripper_mover_rot_mat = quats_to_rot_matrices(self.gripper_mover_rot)
        mover_finger_offset = torch.matmul(gripper_mover_rot_mat, self._mover_finger_vector)
        self.mover_pinch_pos = self.gripper_mover_pos[:, :] + mover_finger_offset
        
        gripper_stator_pos, gripper_stator_rot = self._z1s._gripperStator.get_world_poses(clone=False)
        gripper_stator_pos = gripper_stator_pos - self._env_pos
        self.gripper_stator_pos = torch.tensor(gripper_stator_pos, dtype=torch.float32)
        self.gripper_stator_rot = torch.tensor(gripper_stator_rot, dtype=torch.float32)
        
        gripper_stator_rot_mat = quats_to_rot_matrices(self.gripper_stator_rot)
        stator_finger_offset = torch.matmul(gripper_stator_rot_mat, self._stator_finger_vector)
        self.stator_pinch_pos = self.gripper_stator_pos[:, :] + stator_finger_offset
        
        self.pinch_center = (self.mover_pinch_pos + self.stator_pinch_pos) / 2.0
        
        self.pinch_to_target = self.pinch_center - self.ball_pos
        
        self.pinch_ball_distance = torch.sqrt(torch.sum((self.pinch_center - self.ball_pos) ** 2, dim=1))
        pinch_ball_distance_obs = self.pinch_ball_distance.view(self.num_envs, 1)
        
        self._success_idices = torch.where(self.pinch_ball_distance < self._success_threshold, torch.ones_like(self._success_idices), self._success_idices)
        
        
        self.obs_buf = torch.cat(
            (           
                pinch_ball_distance_obs,  # 1 added   
                
                scaled_dof_pos,
                scaled_dof_vel, 
                # self.z1_dof_pos,        # 7
                # self.z1_dof_vel,        # 7
                        
                self.gripper_mover_pos, # 3
                self.gripper_mover_rot, # 4
                
                # self.pinch_center,
                # self.ball_pos,
                
                self.pinch_to_target    # 3
                
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
        self.Kp_scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2], device='cuda:0') # downsize the gripper gain
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

        
    def generate_random_points(self, n_points):
        points = []
        while len(points) < n_points:
            point = self.outer_box_size * (torch.rand(2, device=self._device) - 0.5)
            if torch.abs(point[0]) > self.inner_box_size or torch.abs(point[1]) > self.inner_box_size:
                points.append(point)

        return torch.stack(points)
    
    def success_rate_calculator(self, success):
        
        if success:
            self.success_list.append(1)
        else:
            self.success_list.append(0)
            
        if len(self.success_list) >= self._max_success_num:
            success_rate = sum(self.success_list) / self._max_success_num * 100
            
            if success_rate > self.max_success_rate:
                self.max_success_rate = success_rate
            
            print("success rate : ", success_rate, " % / max : ", self.max_success_rate, " %")
            self.success_list.pop(0)
            
            


    def reset_idx(self, env_ids):
        
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)
        
        # RESET Robot
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
        
        self._z1s.set_joint_positions(dof_pos, indices=indices)         # Teleport
        self._z1s.set_joint_velocities(dof_vel, indices=indices)
        
        
        # RESET Ball
        new_ball_pos = self.initial_ball_pos[env_ids]
        random_positions = self.generate_random_points(num_indices)
        
        new_ball_pos[:, 0] += random_positions[:, 0]
        new_ball_pos[:, 1] += random_positions[:, 1]

        random_z = torch.rand(num_indices, device=self._device)
        new_ball_pos[:, 2] += self.height_range * random_z - 0.5 * self.height_range
        
        
        ball_pos = torch.zeros((num_indices, 3), device=self._device)
        ball_pos[:, :] = new_ball_pos
        self._balls.set_world_poses(positions=ball_pos, indices=indices)
        
        for i in env_ids:
            if self._success_idices[i] == 1:
                self.success_rate_calculator(True)
            else:
                self.success_rate_calculator(False)
                
        
        self._success_idices[env_ids] = 0
        
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
        
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()
        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)


    def calculate_metrics(self) -> None:

        self.rew_buf[:] = self.compute_z1_reward()

    def is_done(self) -> None:
        
        low_offset = 0.003
        
        gripper_too_low_condition = (self.gripper_mover_pos[:, 2] < self._gripper_too_low_height - low_offset)
        self.reset_buf = torch.where(gripper_too_low_condition, torch.ones_like(self.reset_buf), self.reset_buf)         # on the ground
        
        # reset when the episode reach max
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, 
                                     torch.ones_like(self.reset_buf), self.reset_buf)
    
    def compute_z1_reward(self):

        # distance reward
        # pinch_ball_distance = torch.sqrt(torch.sum((self.pinch_center - self.ball_pos) ** 2, dim=1))
        
        # tried : 10.0, 2.5, 5.0, 7.5
        self.distance_sharpness = 3.5
        rewards = 10.0 * (1.0 - torch.tanh(self.distance_sharpness * self.pinch_ball_distance) ** 2)
        
        # Success reward
        # rewards = torch.where(pinch_ball_distance < self._success_threshold, 10.0, rewards)
        rewards = torch.where(self.pinch_ball_distance < self._success_threshold, rewards + 125.0, rewards)
        
        # jitter penalize
        jitter_penalty_scale = 0.01
        action_diff = self.current_actions - self.prev_actions
        sum_abs_action_diff = torch.sum(abs(action_diff), dim=1)
        rewards = rewards - jitter_penalty_scale * sum_abs_action_diff
        
        gripper_too_low_condition = (self.gripper_mover_pos[:, 2] < self._gripper_too_low_height).unsqueeze(0)
        rewards = torch.where(gripper_too_low_condition, torch.ones_like(rewards) * -10.0, rewards)
         
        # print("[Rewards] : ", rewards)
        
        return rewards
    