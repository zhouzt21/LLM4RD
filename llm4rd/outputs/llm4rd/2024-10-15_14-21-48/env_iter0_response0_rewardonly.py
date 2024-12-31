@torch.jit.script
def compute_reward(self_root_state, self_body_pos, oppo_root_state, oppo_body_pos, oppo_body_rot,
                 oppo_dof_pos, oppo_dof_vel, self_contact_norm, oppo_contact_norm,
                 hand_ids, target_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]

    reward = 0.0
    rewards_dict = {}

    # distance reward for delivering effective blows
    dist_reward = torch.zeros_like(self_root_state[:, 0])
    tar_hand_pos_diff = self_body_pos[:, hand_ids, :].unsqueeze(-2) - oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    local_tar_hand_pos_diff = torch_utils.quat_rotate(heading_rot.unsqueeze(-2).repeat((1, tar_hand_pos_diff.shape[0] // heading_rot.shape[0], 1)), tar_hand_pos_diff)
    dist_reward += (local_tar_hand_pos_diff.norm(dim=-1) > 0.5).float() * 0.05

    # knock out reward
    knock_out_reward = torch.zeros_like(self_root_state[:, 0])
    if oppo_contact_norm[0, 0] > 10:
        knock_out_reward += 100.0

    # time penalty
    time_penalty = self_root_state[:, 6]
    rewards_dict['dist_reward'] = dist_reward * (time_penalty < 300).float()
    rewards_dict['knock_out_reward'] = knock_out_reward * (time_penalty < 300).float() 
    reward = dist_reward + knock_out_reward - time_penalty
    return reward, rewards_dict
