@torch.jit.script
def compute_boxing_observation(self_root_state, self_body_pos, oppo_root_state, oppo_body_pos, oppo_body_rot,
                                  oppo_dof_pos, oppo_dof_vel, self_contact_norm, oppo_contact_norm,
                                  hand_ids, target_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    # root info
    self_root_pos = self_root_state[:, 0:3]
    self_root_rot = self_root_state[:, 3:7]
    oppo_root_pos = oppo_root_state[:, 0:3]
    oppo_root_rot = oppo_root_state[:, 3:7]
    oppo_root_vel = oppo_root_state[:, 7:10]
    oppo_root_ang_vel = oppo_root_state[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(self_root_rot)
    root_pos_diff = oppo_root_pos - self_root_pos
    root_pos_diff[..., -1] = oppo_root_pos[..., -1]
    local_root_pos_diff = torch_utils.quat_rotate(heading_rot, root_pos_diff)

    local_tar_rot = torch_utils.quat_mul(heading_rot, oppo_root_rot)
    local_root_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    local_oppo_vel = torch_utils.quat_rotate(heading_rot, oppo_root_vel)
    local_oppo_ang_vel = torch_utils.quat_rotate(heading_rot, oppo_root_ang_vel)

    oppo_dof_obs = dof_to_obs_smpl(oppo_dof_pos)
    oppo_body_pos_diff = oppo_body_pos - self_root_pos[:, None]
    flat_oppo_body_pos_diff = oppo_body_pos_diff.view(oppo_body_pos_diff.shape[0] * oppo_body_pos_diff.shape[1],
                                              oppo_body_pos_diff.shape[2])
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, oppo_body_pos_diff.shape[1], 1))

    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_flat_oppo_body_pos_diff =  torch_utils.quat_rotate(flat_heading_rot, flat_oppo_body_pos_diff)
    local_flat_oppo_body_pos_diff = local_flat_oppo_body_pos_diff.view(oppo_body_pos_diff.shape[0], oppo_body_pos_diff.shape[1] * oppo_body_pos_diff.shape[2])

    flat_oppo_body_rot = oppo_body_rot.view(oppo_body_rot.shape[0] * oppo_body_rot.shape[1], oppo_body_rot.shape[2])
    local_flat_oppo_body_rot = torch_utils.quat_mul(flat_heading_rot, flat_oppo_body_rot)
    local_flat_oppo_body_rot = torch_utils.quat_to_tan_norm(local_flat_oppo_body_rot)
    local_flat_oppo_body_rot = local_flat_oppo_body_rot.view(oppo_body_rot.shape[0], oppo_body_rot.shape[1] * 6)


    self_hand_pos = self_body_pos[:, hand_ids, :].unsqueeze(-2)
    oppo_target_pos = oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot = heading_rot.unsqueeze(-2).\
        repeat((1, global_target_hand_pos_diff.shape[0] // heading_rot.shape[0], 1))
    local_target_hand_pos_diff = torch_utils.quat_rotate(flat_heading_rot.view(-1, 4), global_target_hand_pos_diff)
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot.shape[0], -1)

    obs = torch.cat([local_root_pos_diff, local_root_rot_obs, local_oppo_vel, local_oppo_ang_vel,
                     oppo_dof_obs, oppo_dof_vel, local_flat_oppo_body_pos_diff, local_flat_oppo_body_rot,
                     local_target_hand_pos_diff, self_contact_norm, oppo_contact_norm], dim=-1)
    return obs
