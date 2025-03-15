from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class Go2MYRobot(LeggedRobot):
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        return noise_vec
    
    def compute_observations(self):
        phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,                     # 3
                self.base_ang_vel * self.obs_scales.ang_vel,                     # 3
                self.projected_gravity,                                          # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                self.dof_vel * self.obs_scales.dof_vel,                          # 12
                self.actions,                                                    # 12
                self.last_actions,                                               # 12
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(phase / 2),
                torch.cos(phase / 2),
                torch.sin(phase / 4),
                torch.cos(phase / 4),
            ),
            axis=-1,
        )

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def check_termination(self):
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
    
    def check_termination(self):
        self.reset_buf = (
            self.episode_length_buf > self.max_episode_length
        )

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        return super()._post_physics_step_callback()
    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = quat_from_angle_axis(4 * phase * torch.pi,
                                             torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))
        self.base_init_quat = torch.tensor([0, 0, 0, 1], device=self.device)
        desired_base_quat = quat_mul(quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = inv_quat(desired_base_quat)
        desired_projected_gravity = transform_by_quat(self.global_gravity, inv_desired_base_quat)

        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff
    
    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, 2].clamp(max=3)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_height_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        target_height = 0.3
        height_diff = torch.square(target_height - self.base_pos[:, 2]) * torch.logical_or(current_time < 0.4, current_time > 1.4)
        return height_diff

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff
    
    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        current_time = self.episode_length_buf * self.dt
        cur_footsteps_translated = self.feet_pos - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        
        return stance_diff

    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        foot_height = (self.feet_pos[:, :, 2]).view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)
