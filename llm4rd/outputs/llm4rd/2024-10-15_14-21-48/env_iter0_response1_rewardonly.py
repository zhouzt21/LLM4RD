@torch.jit.script
def compute_reward(self_contact_norm, oppo_dof_pos):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, Tensor]]

    self_contact = self_contact_norm > 0.5
    knock_out = torch.sum(oppo_dof_pos[:, :3] < -0.8).float()

    reward = torch.where(self_contact & knock_out, 10.0, torch.where(self_contact, 1.0, torch.zeros_like(knock_out)))
    component_reward = {'knockout': knock_out}
    
    return reward, component_reward
