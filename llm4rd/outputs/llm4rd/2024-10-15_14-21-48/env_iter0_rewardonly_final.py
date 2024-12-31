def muay_thai_reward(self_root_state, oppo_root_state):
    # Extract necessary information from states
    heading_rot = self_root_state[:, 3:7]
    local_target_hand_pos_diff = self_root_state[:, 7:10] - self_root_state[:, 0:3]
    local_oppo_hand_pos_diff = oppo_root_state[:, 7:10] - oppo_root_state[:, 0:3]
    hand_contact_norm = torch.norm(self_root_state[:, 10:14], dim=-1)
    oppo_hand_contact_norm = torch.norm(oppo_root_state[:, 10:14], dim=-1)

    # Normalize the hand contact norms to the range [0, 100.0] for better reward shaping
    self_contact_norm = (hand_contact_norm / hand_contact_norm.max()) * 100.0
    oppo_contact_norm = (oppo_hand_contact_norm / oppo_hand_contact_norm.max()) * 100.0

    # Calculate the distance and rotation between hands
    dist_diff = torch.norm(local_target_hand_pos_diff - local_oppo_hand_pos_diff, dim=-1).unsqueeze(-1) * 100.0

    hand_rotation_diff = torch_utils.quat_to_tan_norm(torch_utils.quat_mul(heading_rot, oppo_root_state[:, 3:7])) - \
                       torch_utils.quat_to_tan_norm(self_root_state[:, 3:7])

    # Calculate the time left
    time_left = self_contact_norm[0] / 100.0

    # Create a dictionary of individual reward components
    dist_reward = torch.sigmoid(dist_diff) * time_left
    rotation_reward = hand_rotation_diff * torch.sigmoid(-dist_diff)
    contact_reward = oppo_hand_contact_norm - self_contact_norm

    total_reward = dist_reward + rotation_reward + contact_reward

    return total_reward, {
        'distance': dist_reward,
        'rotation': rotation_reward,
        'contact': contact_reward
    }
