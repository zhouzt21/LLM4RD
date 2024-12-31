@torch.jit.script
def compute_reward(self_root_state, self_body_pos, oppo_root_state, oppo_body_pos, oppo_dof_pos,
                  self_contact_norm, oppo_contact_norm, hand_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Dict[str, Tensor]]

    root_info = self_root_state[:, 0:3] - oppo_root_state[:, 0:3]
    heading_rot = torch_utils.calc_heading_quat_inv(self_root_state[:, 3:7])
    local_root_pos_diff = torch_utils.quat_rotate(heading_rot, root_info)
    local_oppo_vel = torch_utils.quat_rotate(heading_rot, oppo_root_state[:, 10:13])

    # hand positions
    self_hand_pos = self_body_pos[:, hand_ids, :].unsqueeze(-2)
    oppo_target_pos = oppo_body_pos.unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot = heading_rot.unsqueeze(-2).\
        repeat((1, global_target_hand_pos_diff.shape[0] // heading_rot.shape[0], 1))
    local_target_hand_pos_diff = torch_utils.quat_rotate(flat_heading_rot.view(-1, 4), global_target_hand_pos_diff)
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot.shape[0], -1)

    # calculate the distance and rotation between hands
    dist_diff = torch.norm(local_target_hand_pos_diff, dim=-1).unsqueeze(-1) * 100.0

    hand_rotation_diff = torch_utils.quat_to_tan_norm(torch_utils.quat_mul(heading_rot, oppo_root_state[:, 3:7])) - \
                       torch_utils.quat_to_tan_norm(self_root_state[:, 3:7])

    # calculate the time left
    time_left = self_contact_norm[0] / 100.0

    # create a dictionary of individual reward components
    dist_reward = torch.sigmoid(dist_diff) * time_left
    rotation_reward = hand_rotation_diff * torch.sigmoid(-dist_diff)
    contact_reward = oppo_contact_norm[0].unsqueeze(-1)

    total_reward = dist_reward + rotation_reward + contact_reward

    return total_reward, {
        'distance': dist_reward,
        'rotation': rotation_reward,
        'contact': contact_reward
    }
