import torch
from typing import Dict, Tuple

def calculate_rewards_and_dones_jit(
    # 包含了所有超参数的字典，方便传入JIT函数
    h_params: Dict[str, float],
    # 包含世界状态的张量
    all_pos: torch.Tensor,
    all_vel: torch.Tensor,
    p_vels: torch.Tensor,
    p_raw_actions: torch.Tensor,
    raw_actions: torch.Tensor,
    basket_pos: torch.Tensor,
    spot_center_pos: torch.Tensor,
    t_remaining: torch.Tensor,
    # 需要在函数内部更新并返回的状态张量
    a1_still_frames_counter: torch.Tensor,
    wall_collision_counters: torch.Tensor,
    defender_over_midline_counter: torch.Tensor,
    termination_reason_code: torch.Tensor,
    dones: torch.Tensor,
    # 预先计算好的交互张量
    dist_matrix: torch.Tensor,
    collision_matrix: torch.Tensor,
    vel_diffs_norm: torch.Tensor,
    requested_accelerations_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT兼容的全向量化函数，用于并行计算奖励和回合终止信号。
    此版本移除了稠密奖励部分针对智能体的for循环，以实现最大程度的并行化。

    返回:
        dense_reward (Tensor): 所有智能体在当前步的稠密奖励
        terminal_rewards (Tensor): 所有智能体的终局奖励 (仅在回合结束时非零)
        dones_out (Tensor): 所有环境的终止标志
        a1_still_frames_counter (Tensor): 更新后的A1静止计数器
        wall_collision_counters (Tensor): 更新后的撞墙计数器
        defender_over_midline_counter (Tensor): 更新后的防守方越线计数器
        shots_this_step (Tensor): 标记哪些环境在当前步发生了投篮
    """
    # 从输入中提取维度和常量
    batch_dim, n_agents, _ = all_pos.shape
    device = all_pos.device
    n_attackers = 2
    n_defenders = 2
    
    # 初始化返回的张量
    terminal_rewards = torch.zeros(batch_dim, n_agents, device=device)
    dense_reward = torch.zeros(batch_dim, n_agents, device=device)
    dones_out = dones.clone()
    attacker_win_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)
    reason_code = termination_reason_code.clone()

    # =================================================================================
    # 第一部分: 回合终止条件检查
    # =================================================================================

    # --- 条件1: 尝试投篮 (Shot Attempt) ---
    a1_pos = all_pos[:, 0]
    a1_vel = all_vel[:, 0]
    dist_to_spot = torch.linalg.norm(a1_pos - spot_center_pos, dim=1)
    in_area = (dist_to_spot <= h_params['R_spot']) & (a1_pos[:, 1] > 0)
    is_still = torch.linalg.norm(a1_vel, dim=1) < h_params['v_shot_threshold']
    not_accelerating = torch.linalg.norm(raw_actions[:, 0, :], dim=1) < h_params['a_shot_threshold']
    is_ready_to_shoot = in_area & is_still & not_accelerating
    prev_still_counter = a1_still_frames_counter
    curr_still_counter = torch.where(is_ready_to_shoot, prev_still_counter + 1, 0)
    
    shot_attempted = (curr_still_counter >= h_params['shot_still_frames']) & ~dones_out
    if torch.any(shot_attempted):
        # attacker_win_this_step |= shot_attempted
        shot_b_idx = shot_attempted.nonzero().squeeze(-1)
        
        a1_pos_shot = a1_pos[shot_b_idx]
        a2_pos_shot = all_pos[shot_b_idx, 1]
        spot_pos_shot = spot_center_pos[shot_b_idx]
        basket_pos_shot = basket_pos[shot_b_idx]
        defender_pos_shot = all_pos[shot_b_idx][:, 2:]

        shot_vector = basket_pos_shot - a1_pos_shot
        blocker_vector = defender_pos_shot - a1_pos_shot.unsqueeze(1)
        shot_vector_norm_sq = torch.sum(shot_vector**2, dim=-1, keepdim=True) + 1e-6
        dot_product = torch.sum(blocker_vector * shot_vector.unsqueeze(1), dim=-1)
        proj_len_ratio = dot_product / shot_vector_norm_sq
        is_between = (proj_len_ratio > 0) & (proj_len_ratio < 1)
        projection = proj_len_ratio.unsqueeze(-1) * shot_vector.unsqueeze(1)
        dist_perp_sq = torch.sum((blocker_vector - projection)**2, dim=-1)
        dist_a1_to_def = torch.linalg.norm(blocker_vector, dim=-1)
        gate_input = h_params['def_proximity_threshold'] - dist_a1_to_def
        soft_proximity_gate = torch.sigmoid(h_params['block_gate_k'] * gate_input)
        is_blocker_per_defender = is_between & (dist_perp_sq < (h_params['proximity_threshold'])**2)
        block_contribution = (
            torch.exp(-dist_perp_sq / (2 * h_params['block_sigma']**2)) # 基于横向距离的贡献
            * is_blocker_per_defender.float()                          # 基于站位的硬门控
            * soft_proximity_gate                                      # 基于纵向距离的软门控 (新！)
        )
        total_block_factor = torch.clamp(block_contribution.sum(dim=1), 0, 1)

        is_a_winning_shot = total_block_factor < h_params['win_condition_block_threshold']
        winning_shot_indices = shot_b_idx[is_a_winning_shot]
        losing_shot_indices = shot_b_idx[~is_a_winning_shot]

        if winning_shot_indices.numel() > 0:
            attacker_win_this_step[winning_shot_indices] = True
            reason_code[winning_shot_indices] = 1 # 原因码1: 投篮命中

        if losing_shot_indices.numel() > 0:
            reason_code[losing_shot_indices] = 11 # 原因码11: 投篮被盖
        
        base_score = h_params['max_score'] * (1 - dist_to_spot[shot_b_idx] / h_params['R_spot']) + h_params['shoot_score']
        final_score_modified = base_score * (1 - total_block_factor)
        time_bonus = h_params['k_time_bonus'] * (t_remaining[shot_b_idx].squeeze(-1) / h_params['t_limit'])
        dist_a1_to_defs = torch.linalg.norm(blocker_vector, dim=-1)
        avg_dist_to_defs = torch.mean(dist_a1_to_defs, dim=1)
        spacing_bonus = h_params['k_spacing_bonus'] * avg_dist_to_defs
        a1_reward = final_score_modified + spacing_bonus + time_bonus
        terminal_rewards[shot_b_idx, 0] += a1_reward

        dist_a1_to_defs_sq = torch.sum(blocker_vector**2, dim=-1)
        _, closest_def_indices = torch.min(dist_a1_to_defs_sq, dim=1)
        batch_indices = torch.arange(len(shot_b_idx), device=device)
        p_closest_def = defender_pos_shot[batch_indices, closest_def_indices]
        
        def_to_a1_vec = a1_pos_shot - p_closest_def
        def_to_a1_norm = torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6
        def_to_a1_unit_vec = def_to_a1_vec / def_to_a1_norm
        ideal_screen_pos = p_closest_def + h_params['screen_pos_offset'] * def_to_a1_unit_vec
        
        dist_a2_to_ideal_sq = torch.sum((a2_pos_shot - ideal_screen_pos)**2, dim=-1)
        vec_a2_to_def = p_closest_def - a2_pos_shot
        vec_a2_to_a1 = a1_pos_shot - a2_pos_shot
        dot_product_gate = torch.sum(vec_a2_to_def * vec_a2_to_a1, dim=-1)
        screen_gate = torch.sigmoid(-h_params['k_screen_gate'] * dot_product_gate)
        screen_bonus = h_params['k_a2_screen_bonus'] * torch.exp(-dist_a2_to_ideal_sq / (2 * h_params['a2_screen_sigma']**2)) * screen_gate
        a2_reward = final_score_modified + screen_bonus + spacing_bonus + time_bonus
        terminal_rewards[shot_b_idx, 1] += a2_reward

        for i in range(n_defenders):
            R_block = h_params['k_def_block_reward'] * block_contribution[:, i]
            dist_a1_to_spot_shot = dist_to_spot[shot_b_idx]
            R_force = h_params['k_def_force_reward'] * (dist_a1_to_spot_shot / h_params['R_spot'])
            
            a1_to_spot_unit_vec = (basket_pos_shot - a1_pos_shot) / (torch.linalg.norm(basket_pos_shot - a1_pos_shot, dim=-1, keepdim=True) + 1e-6)
            d_from_a1_vec = defender_pos_shot[:, i, :] - a1_pos_shot
            proj_dot = torch.sum(d_from_a1_vec * a1_to_spot_unit_vec, dim=-1)
            pos_gate = torch.sigmoid(5.0 * proj_dot)
            ideal_pos = a1_pos_shot + h_params['def_pos_offset'] * a1_to_spot_unit_vec
            dist_to_ideal_sq = torch.sum((defender_pos_shot[:, i, :] - ideal_pos)**2, dim=-1)
            positioning_reward_factor = torch.exp(-dist_to_ideal_sq / (2 * h_params['def_pos_sigma']**2))
            R_positioning = h_params['k_def_pos_reward'] * positioning_reward_factor * pos_gate
            
            dist_def_to_spot_sq = torch.sum((defender_pos_shot[:, i, :] - spot_pos_shot)**2, dim=-1)
            R_area_control = h_params['k_def_area_reward'] * torch.exp(-dist_def_to_spot_sq / (2 * h_params['def_gaussian_spot_sigma']**2))
            
            total_def_reward = R_block + R_force + R_positioning + R_area_control - h_params['k_def_shot_penalty']
            terminal_rewards[shot_b_idx, n_attackers + i] += total_def_reward
        
        dones_out |= shot_attempted

    # --- 条件2: 时间耗尽 (Time Up) ---
    time_up = (t_remaining.squeeze(-1) <= 0) & ~dones_out
    if torch.any(time_up):
        a1_pos_timeout = a1_pos[time_up]
        dist_a1_to_spot_timeout = torch.linalg.norm(a1_pos_timeout - spot_center_pos[time_up], dim=-1)
        attacker_penalty = -h_params['attacker_timeout_penalty'] - h_params['k_timeout_dist_penalty'] * dist_a1_to_spot_timeout
        attacker_penalty_clamped = torch.clamp(attacker_penalty,0,h_params['attacker_timeout_penalty_max'])
        terminal_rewards[time_up, 0] = attacker_penalty_clamped
        terminal_rewards[time_up, 1] = h_params["foul_teammate_factor"] * attacker_penalty_clamped
        terminal_rewards[time_up, n_attackers:] = -attacker_penalty_clamped.unsqueeze(-1)
        reason_code[time_up] = 12 # 原因码12: 进攻超时
        dones_out |= time_up
    
    # --- 条件3: 碰撞犯规 (Foul) ---
    is_foul = collision_matrix & (vel_diffs_norm > h_params['v_foul_threshold']) & ~dones_out.view(-1, 1, 1)
    if torch.triu(is_foul, diagonal=1).any():
        foul_indices = torch.triu(is_foul, diagonal=1).nonzero()
        
        # JIT不支持张量解包 (JIT does not support tensor unpacking)
        # b_idx, i_idx, j_idx = foul_indices.T
        # --- JIT-compatible fix: ---
        b_idx = foul_indices[:, 0]
        i_idx = foul_indices[:, 1]
        j_idx = foul_indices[:, 2]

        relative_speeds = vel_diffs_norm[b_idx, i_idx, j_idx]
        dynamic_foul_magnitude = h_params['R_foul'] + h_params['k_foul_vel_penalty'] * relative_speeds
        
        agent_i_p_vel = p_vels[b_idx, i_idx]
        pos_rel = all_pos[b_idx, j_idx] - all_pos[b_idx, i_idx]
        vel_rel_on_pos = torch.einsum("bd,bd->b", agent_i_p_vel, pos_rel)
        i_is_active = vel_rel_on_pos > 0
        active_indices = torch.where(i_is_active, i_idx, j_idx)
        passive_indices = torch.where(i_is_active, j_idx, i_idx)
        
        active_is_attacker = active_indices < n_attackers
        passive_is_attacker = passive_indices < n_attackers
        is_friendly_fire = (active_is_attacker == passive_is_attacker)
        
        foul_rewards = torch.zeros_like(terminal_rewards)
        opp_foul_mask = ~is_friendly_fire
        if torch.any(opp_foul_mask):
            opp_b = b_idx[opp_foul_mask]
            opp_active = active_indices[opp_foul_mask]
            opp_passive = passive_indices[opp_foul_mask]
            opp_magnitude = dynamic_foul_magnitude[opp_foul_mask]
            
            # teammate_map = torch.tensor([1, 0, 3, 2], device=device, dtype=torch.long)
            # opp_active_teammate = teammate_map[opp_active]
            # opp_passive_teammate = teammate_map[opp_passive]
            
            num_opp_fouls = opp_b.shape[0]
            opp_rewards_to_add = torch.zeros(num_opp_fouls, n_agents, device=device)
            opp_row_indices = torch.arange(num_opp_fouls, device=device)

            opp_rewards_to_add[opp_row_indices, opp_active] = -opp_magnitude
            opp_rewards_to_add[opp_row_indices, opp_passive] = opp_magnitude * h_params['foul_teammate_factor']
            # opp_rewards_to_add[opp_row_indices, opp_active_teammate] = -opp_magnitude * h_params['foul_teammate_factor']
            # opp_rewards_to_add[opp_row_indices, opp_passive_teammate] = opp_magnitude * h_params['foul_teammate_factor']
            
            opp_active_is_defender = opp_active >= n_attackers
            b_idx_def_foul = opp_b[opp_active_is_defender]
            attacker_win_this_step[b_idx_def_foul] = True
            reason_code[b_idx_def_foul] = 2 # 原因码2: 对手犯规
            foul_rewards.index_add_(0, opp_b, opp_rewards_to_add)
            b_idx_att_foul = opp_b[~opp_active_is_defender]
            reason_code[b_idx_att_foul] = 13 # 原因码13: 己方犯规

        ff_foul_mask = is_friendly_fire
        if torch.any(ff_foul_mask):
            ff_b = b_idx[ff_foul_mask]
            ff_active = active_indices[ff_foul_mask]
            ff_passive = passive_indices[ff_foul_mask]
            ff_magnitude = dynamic_foul_magnitude[ff_foul_mask]

            num_ff_fouls = ff_b.shape[0]
            ff_rewards_to_add = torch.zeros(num_ff_fouls, n_agents, device=device)
            ff_row_indices = torch.arange(num_ff_fouls, device=device)

            ff_rewards_to_add[ff_row_indices, ff_active] = -ff_magnitude
            ff_rewards_to_add[ff_row_indices, ff_passive] = -ff_magnitude
            
            ff_active_is_attacker = active_is_attacker[ff_foul_mask]
            # 防守方误伤 -> 进攻方胜利
            b_idx_def_ff = ff_b[~ff_active_is_attacker]
            attacker_win_this_step[b_idx_def_ff] = True
            reason_code[b_idx_def_ff] = 5 # 原因码5: 对手友军误伤

            # 进攻方误伤 -> 进攻方失败
            b_idx_att_ff = ff_b[ff_active_is_attacker]
            reason_code[b_idx_att_ff] = 15 # 原因码15: 己方友军误伤
            
            foul_rewards.index_add_(0, ff_b, ff_rewards_to_add)

        terminal_rewards += foul_rewards
        dones_out[b_idx] = True

    # --- 条件4: 持续撞墙导致超时 (Wall Collision Timeout) ---
    is_wall_timeout = (wall_collision_counters >= h_params['wall_collision_frames']) & ~dones_out.view(-1, 1)
    if torch.any(is_wall_timeout):
        wall_indices = is_wall_timeout.nonzero()
        
        # JIT不支持张量解包 (JIT does not support tensor unpacking)
        # b_idx, agent_idx = wall_indices.T
        # --- JIT-compatible fix: ---
        b_idx = wall_indices[:, 0]
        agent_idx = wall_indices[:, 1]
        
        is_defender_wall_collision = agent_idx >= n_attackers
        b_idx_def_wall_coll = b_idx[is_defender_wall_collision]
        b_idx_att_wall_coll = b_idx[~is_defender_wall_collision]
        attacker_win_this_step[b_idx_def_wall_coll] = True
        reason_code[b_idx_def_wall_coll] = 3 # 原因码3: 对手失误-撞墙
        reason_code[b_idx_att_wall_coll] = 14 # 原因码14: 己方失误-撞墙
        terminal_rewards[b_idx, agent_idx] += h_params['R_wall_collision_penalty']
        dones_out[b_idx] = True

    # --- 条件5: 防守方越过中线过久犯规 (Defender Over Midline Foul) ---
    defender_pos = all_pos[:, n_attackers:]
    is_over_midline = defender_pos[:, :, 1] < 0
    defender_over_midline_counter = torch.where(is_over_midline, defender_over_midline_counter + 1, 0)

    midline_foul_triggered = (defender_over_midline_counter >= h_params['max_time_over_midline']) & ~dones_out.view(-1, 1)
    if torch.any(midline_foul_triggered):
        foul_indices = midline_foul_triggered.nonzero()
        b_idx = foul_indices[:, 0]
        
        attacker_win_this_step[b_idx] = True
        reason_code[b_idx] = 4 # 原因码4: 对手失误-越线
        terminal_rewards[b_idx, n_attackers:] -= h_params['R_midline_foul']
        dones_out[b_idx] = True


    # =================================================================================
    # 第二部分: 稠密奖励计算 (完全向量化的版本)
    # =================================================================================
    
    # --- 1. 创建角色掩码 ---
    agent_indices = torch.arange(n_agents, device=device)
    a1_mask = (agent_indices == 0).view(1, -1)
    a2_mask = (agent_indices == 1).view(1, -1)
    attacker_mask = (agent_indices < n_attackers).view(1, -1)
    defender_mask = (agent_indices >= n_attackers).view(1, -1)

    # --- 2. 通用奖励/惩罚 (对所有智能体生效) ---

    # 2.1 出界惩罚
    safe_x = h_params['W'] / 2 - (h_params['agent_radius'] / 2)
    safe_y = h_params['L'] / 2 - (h_params['agent_radius'] / 2)
    oob_depth_x = torch.logaddexp(torch.tensor(0.0, device=device), (torch.abs(all_pos[..., 0]) - safe_x) / h_params['oob_margin'])
    oob_depth_y = torch.logaddexp(torch.tensor(0.0, device=device), (torch.abs(all_pos[..., 1]) - safe_y) / h_params['oob_margin'])
    velocity_norm = torch.linalg.norm(all_vel, dim=-1) + 1.0
    oob_reward = h_params['oob_penalty'] * h_params['oob_margin'] * (oob_depth_x + oob_depth_y) * velocity_norm
    dense_reward += oob_reward

    # 2.2 控制量惩罚
    raw_u_norm = torch.linalg.vector_norm(raw_actions, dim=-1)
    dense_reward -= h_params['k_u_penalty_general'] * raw_u_norm

    requested_a_norm = torch.linalg.norm(requested_accelerations_tensor, dim=-1)
    excess_acceleration = torch.clamp(requested_a_norm - h_params['a_max'], min=0.0)
    acceleration_penalty = -h_params['k_excess_acceleration_penalty'] * (excess_acceleration ** 2)
    dense_reward += acceleration_penalty

    # 2.3 动作平滑度惩罚 (Jerk Penalty)
    action_jerk = torch.linalg.norm(raw_actions - p_raw_actions, dim=-1)
    jerk_penalty = -h_params['k_action_jerk_penalty'] * action_jerk
    dense_reward += jerk_penalty

    # --- 3. 基于距离和碰撞的复杂通用惩罚 ---

    # 3.1 近距离惩罚 (最终修复版)
    dist_matrix_no_self = dist_matrix.clone()
    dist_matrix_no_self.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
    
    # --- 步骤1: 向量化计算每个智能体专属的惩罚参数 ---
    
    # a. 惩罚系数 (k_prox)
    dist_d_to_spot = torch.linalg.norm(all_pos - spot_center_pos.unsqueeze(1), dim=-1)
    is_in_spot_area = (dist_d_to_spot <= h_params['R_spot'])
    k_def_proximity = torch.where(
        is_in_spot_area, 
        h_params['k_def_proximity_penalty'] * (1 - h_params['proximity_penalty_reduction_in_spot']), 
        h_params['k_def_proximity_penalty']
    )
    k_prox = torch.zeros_like(dense_reward)
    k_prox += h_params['k_a1_proximity_penalty'] * a1_mask
    k_prox += h_params['k_proximity_penalty'] * a2_mask
    k_prox += k_def_proximity * defender_mask
    
    # b. 触发距离 (prox_threshold)
    prox_threshold = torch.where(
        a1_mask, 
        h_params['a1_proximity_threshold'], 
        h_params['proximity_threshold']
    )
    
    # c. [新] 惩罚软度 (k_margin)，为A1做特殊处理
    k_margin_per_agent = torch.where(
        a1_mask,
        h_params['a1_proximity_penalty_margin'], # A1专属的软度参数
        h_params['proximity_penalty_margin']      # 其他智能体通用的软度参数
    )

    # --- 步骤2: 计算并应用惩罚 ---
    
    is_too_close = dist_matrix_no_self < prox_threshold.unsqueeze(-1)
    
    if torch.any(is_too_close):
        # 扩展维度以用于广播计算 (B, N, N)
        k_margin_tensor = k_margin_per_agent.unsqueeze(-1)
        prox_threshold_tensor = prox_threshold.unsqueeze(-1)
        
        # 使用每个智能体专属的软度参数 k_margin_tensor 来计算穿透深度
        penetration = torch.logaddexp(
            torch.tensor(0.0, device=device), 
            (prox_threshold_tensor - dist_matrix_no_self) / k_margin_tensor
        ) * k_margin_tensor
        
        # 计算最终惩罚值
        proximity_penalty = -k_prox.unsqueeze(-1) * penetration
        
        # 应用 is_too_close 掩码，并将惩罚施加给对应的智能体
        dense_reward += (proximity_penalty * is_too_close.float()).sum(dim=-1)

    # 3.2 碰撞惩罚
    agent_collisions = collision_matrix.diagonal(dim1=-2, dim2=-1).any(-1)
    if torch.any(agent_collisions):
        # 高速碰撞
        pos_rel = all_pos.unsqueeze(2) - all_pos.unsqueeze(1) # B,N,N,2
        vel_proj = torch.einsum("bnd,bnmd->bnm", all_vel, pos_rel)
        is_active = vel_proj > 0
        active_penalty = -h_params['k_coll_active'] * vel_diffs_norm
        passive_penalty = -h_params['k_coll_passive'] * vel_diffs_norm
        penalty = torch.where(is_active, active_penalty, passive_penalty)
        dense_reward += (penalty * collision_matrix.float()).sum(dim=-1)

        # 低速推挤
        is_low_speed_collision = collision_matrix & (vel_diffs_norm < h_params['low_velocity_threshold'])
        if torch.any(is_low_speed_collision):
            push_penalty_coeff = torch.where(attacker_mask, h_params['k_push_penalty'], h_params['k_def_push_penalty'])
            pos_diffs_norm = torch.linalg.norm(pos_rel, dim=-1, keepdim=True) + 1e-6
            proj_vector = pos_rel / pos_diffs_norm
            push_force_magnitude = torch.einsum('bnd,bnmd->bnm', raw_actions, proj_vector)
            push_penalty = -push_penalty_coeff.unsqueeze(-1) * torch.clamp(push_force_magnitude, min=0.0)
            dense_reward += (push_penalty * is_low_speed_collision.float()).sum(dim=-1)
    
    # 3.3 造犯规奖励
    is_standing_still = torch.linalg.norm(all_vel, dim=-1) < h_params['stand_still_threshold'] # B,N
    is_to_stand = raw_u_norm < h_params['stand_still_threshold']
    relative_pos_all = all_pos.unsqueeze(2) - all_pos.unsqueeze(1) # B,N,N,2
    relative_dist_all = torch.linalg.norm(relative_pos_all, dim=-1) # B,N,N
    is_within_charge_range = relative_dist_all < h_params['charge_drawing_range']
    dot_product = torch.sum(all_vel.unsqueeze(1) * relative_pos_all, dim=-1) # B,N,N
    speed_of_approach = dot_product / (relative_dist_all + 1e-6)
    approach_gradient = torch.clamp(speed_of_approach, min=0) # B,N,N
    
    agent_is_attacker = attacker_mask.squeeze(0) # N
    is_opponent_matrix = agent_is_attacker.unsqueeze(1) != agent_is_attacker.unsqueeze(0) # N,N
    
    reward_for_opponents = h_params['k_stand_still_reward'] * approach_gradient * is_standing_still.unsqueeze(-1).float() * is_to_stand.unsqueeze(-1).float() * is_within_charge_range.float() * is_opponent_matrix.float() # B,N,N
    charge_drawing_reward = reward_for_opponents.sum(dim=-1) # B,N
    dense_reward += charge_drawing_reward

    # --- 4. 分角色奖励/惩罚 ---

    # 4.1 A1 (持球进攻者) 奖励/惩罚
    a1_pos = all_pos[:, 0]
    a1_vel = all_vel[:, 0]
    a1_speed = torch.linalg.norm(a1_vel, dim=1)
    raw_a1_u_norm = torch.linalg.vector_norm(raw_actions[:, 0, :], dim=-1)
    dist_a1_to_spot = torch.linalg.norm(a1_pos - spot_center_pos, dim=1)
    is_in_spot_a1 = (dist_a1_to_spot <= h_params['R_spot']) & (a1_pos[:, 1] > 0)

    # --- 基础奖励：引导A1进入投篮区域 ---
    # 1. 向投篮点中心的高斯吸引力奖励
    a1_gaussian_factor = torch.exp(- (dist_a1_to_spot**2) / (2 * h_params['gaussian_sigma']**2))
    gaussian_reward = h_params['gaussian_scale'] * a1_gaussian_factor
    
    # 2. 朝着投篮点移动的速度奖励
    vector_to_spot = spot_center_pos - a1_pos
    vector_to_spot_norm = vector_to_spot / (torch.linalg.norm(vector_to_spot, dim=1, keepdim=True) + 1e-6)
    speed_projection = torch.sum(a1_vel * vector_to_spot_norm, dim=1)
    speed_spot_reward = h_params['k_a1_speed_spot_reward'] * (1 - a1_gaussian_factor)  * speed_projection
    
    # 3. 在投篮区域内的存在奖励
    in_spot_reward = h_params['k_a1_in_spot_reward'] * (1 - dist_a1_to_spot / h_params['R_spot']) * is_in_spot_a1.float()

    # --- 惩罚项：被封盖与犹豫 ---
    # 1. 被封盖惩罚 (采用软门控逻辑)
    def_pos = all_pos[:, n_attackers:]
    ap = basket_pos.unsqueeze(1) - a1_pos.unsqueeze(1)
    ad = def_pos - a1_pos.unsqueeze(1)
    proj_len_ratio_blocked = torch.einsum("bnd,bmd->bnm", ad, ap).squeeze(-1) / (torch.linalg.norm(ap, dim=-1).pow(2) + 1e-6)
    is_between_mask = (proj_len_ratio_blocked > 0) & (proj_len_ratio_blocked < 1)
    closest_point_on_line = a1_pos.unsqueeze(1) + proj_len_ratio_blocked.unsqueeze(-1) * ap
    dist_perp = torch.linalg.norm(def_pos - closest_point_on_line, dim=-1)
    dist_to_def = torch.linalg.norm(ad, dim=-1)
    gate_input_a1 = h_params['def_proximity_threshold'] - dist_to_def
    soft_proximity_gate_a1 = torch.sigmoid(h_params['block_gate_k'] * gate_input_a1)
    block_factor_a1 = (
        torch.exp(-dist_perp.pow(2) / (2 * h_params['block_sigma'] ** 2))
        * is_between_mask.float()
        * soft_proximity_gate_a1
    )
    total_block_factor_a1 = block_factor_a1.sum(dim=1)
    blocked_penalty = total_block_factor_a1 * h_params['k_a1_blocked_penalty']

    # 2. 在非投篮区移动过慢的犹豫惩罚
    hesitation_factor = torch.clamp((1.0 - (a1_speed / h_params['hesitate_speed_threshold'])), min=0.0)**2
    hesitation_penalty = -h_params['k_hesitation_penalty'] * (hesitation_factor) * (~is_in_spot_a1).float()

    # --- 动态行为奖励模块 ---
    # 根据被封盖程度，在“静止”和“移动”两种策略间动态切换

    # 潜力奖励1: “静止奖励” (当未被封锁时应该获得的奖励)
    base_velocity_reward = torch.exp(-(a1_speed**2) / (2 * h_params['velocity_stillness_sigma']**2))
    velocity_stillness_reward = h_params['k_a1_velocity_stillness_reward'] * base_velocity_reward
    is_willing_to_stop = (raw_a1_u_norm < h_params['low_u_threshold'])
    base_action_reward = torch.exp(-(raw_a1_u_norm**2) / (2 * h_params['action_stillness_sigma']**2))
    action_stillness_reward = h_params['k_a1_action_stillness_reward'] * base_action_reward
    final_action_stillness_reward = action_stillness_reward * is_willing_to_stop.float()
    stillness_reward = (velocity_stillness_reward + final_action_stillness_reward) * is_in_spot_a1.float()

    # 潜力奖励2: “移动奖励” (当被封锁时应该获得的奖励)
    dist_a1_to_defs = torch.linalg.norm(a1_pos.unsqueeze(1) - def_pos, dim=-1)
    closest_def_indices = torch.argmin(dist_a1_to_defs, dim=1)
    batch_indices = torch.arange(batch_dim, device=device)
    p_closest_def = def_pos[batch_indices, closest_def_indices]
    vec_def_to_a1 = a1_pos - p_closest_def
    unit_vec_away_from_def = vec_def_to_a1 / (torch.linalg.norm(vec_def_to_a1, dim=-1, keepdim=True) + 1e-6)
    speed_of_separation = torch.sum(a1_vel * unit_vec_away_from_def, dim=1)
    separation_reward = h_params['k_a1_separation_reward'] * torch.clamp(speed_of_separation, min=0.0)

    # 【核心】进行动态加权
    dynamic_behavior_reward = (
        (1.0 - total_block_factor_a1) * stillness_reward + 
        total_block_factor_a1 * separation_reward
    )

    # --- 横向机动奖励 (Tangential Movement Reward) ---
    tangential_reward = torch.zeros_like(a1_speed)
    
    # 【修改】：将压力门控的计算移到前面，因为它现在更复杂
    
    # 1. 计算基础的“防守压力”，基于距离 (逻辑不变)
    #    dist_a1_to_defs 和 p_closest_def 我们在前面已经计算过，直接复用
    dist_to_closest_def = torch.min(dist_a1_to_defs, dim=1).values
    base_pressure_gate = torch.exp(-dist_to_closest_def.pow(2) / (2 * h_params['a1_tangential_pressure_sigma']**2))

    # 2. 【新增】计算“位置门控”，判断防守者是否在A1和投篮点之间
    vec_a1_to_basket = basket_pos - a1_pos
    vec_a1_to_def = p_closest_def - a1_pos
    
    # 通过向量投影判断
    dot_product_for_gate = torch.sum(vec_a1_to_def * vec_a1_to_basket, dim=-1)
    norm_sq_for_gate = torch.sum(vec_a1_to_basket**2, dim=-1) + 1e-6
    proj_ratio = dot_product_for_gate / norm_sq_for_gate
    
    # 只有当投影比例在0到1之间，才算是在中间
    is_def_in_front = (proj_ratio > 0) & (proj_ratio < 1)

    # 3. 【修改】最终的压力门控 = 基础距离门控 * 位置门控
    pressure_gate = base_pressure_gate * is_def_in_front.float()
    
    # 计算速度的垂直分量 (逻辑不变)
    spot_vec_norm = torch.linalg.norm(vec_a1_to_basket, dim=-1, keepdim=True) + 1e-6
    unit_vec_to_basket = vec_a1_to_basket / spot_vec_norm
    vel_parallel_mag = torch.sum(a1_vel * unit_vec_to_basket, dim=-1, keepdim=True)
    vel_parallel_vec = vel_parallel_mag * unit_vec_to_basket
    vel_perp_vec = a1_vel - vel_parallel_vec
    tangential_speed = torch.linalg.norm(vel_perp_vec, dim=-1)
    
    # 计算最终的横向机动奖励
    tangential_reward = h_params['k_a1_tangential_reward'] * tangential_speed * pressure_gate

    # --- 投篮蓄力奖励&放弃惩罚 ---
    ready_to_shoot_reward = h_params['k_a1_ready_to_shoot_reward'] * is_ready_to_shoot.float()
    shot_was_abandoned = (prev_still_counter > 0) & (curr_still_counter == 0)
    
    # 2. 计算惩罚值：惩罚大小与放弃前的计数值成正比
    abandon_shot_penalty = -h_params['k_a1_ready_to_shoot_reward'] * shot_was_abandoned.float()

    # --- 汇总所有A1的稠密奖励 ---
    total_a1_reward = (
        gaussian_reward 
        + speed_spot_reward 
        + in_spot_reward 
        + blocked_penalty
        + hesitation_penalty
        + dynamic_behavior_reward
        + tangential_reward
        + abandon_shot_penalty
        + ready_to_shoot_reward
    )
    dense_reward[:, 0] += total_a1_reward

    # 4.2 A2 (无球掩护者) 奖励/惩罚
    p_a1 = all_pos[:, 0]
    p_a2 = all_pos[:, 1]
    
    dist_a1_to_defs = torch.linalg.norm(p_a1.unsqueeze(1) - def_pos, dim=-1)
    closest_def_indices = torch.argmin(dist_a1_to_defs, dim=1)
    batch_indices = torch.arange(batch_dim, device=device)
    p_closest_def = def_pos[batch_indices, closest_def_indices]
    
    def_to_a1_vec = p_a1 - p_closest_def
    ideal_screen_pos = p_closest_def + h_params['screen_pos_offset'] * (def_to_a1_vec / (torch.linalg.norm(def_to_a1_vec, dim=-1, keepdim=True) + 1e-6))
    dist_a2_to_ideal_sq = torch.sum((p_a2 - ideal_screen_pos)**2, dim=-1)
    vec_a2_to_def = p_closest_def - p_a2
    vec_a2_to_a1 = p_a1 - p_a2
    dot_product_gate = torch.sum(vec_a2_to_def * vec_a2_to_a1, dim=-1)

    # A2是否比靠近A1更靠近防守者 (spacing_gate_factor)
    dist_a2_to_a1 = torch.linalg.norm(vec_a2_to_a1, dim=-1)
    dist_a2_to_def = torch.linalg.norm(vec_a2_to_def, dim=-1)
    # 距离差：如果 > 0，说明离Def更近(好)；如果 < 0，说明离A1更近(坏)
    distance_difference = dist_a2_to_a1 - dist_a2_to_def
    spacing_gate_factor = torch.sigmoid(h_params['screen_spacing_gate_k'] * distance_difference)

    soft_gate_factor = torch.sigmoid(-h_params['k_screen_gate'] * dot_product_gate)
    screen_reward = h_params['k_ideal_screen_pos'] * torch.exp(-dist_a2_to_ideal_sq / (2 * h_params['screen_pos_sigma']**2)) * soft_gate_factor * spacing_gate_factor
    
    shot_vector_a2 = basket_pos - p_a1
    a2_vector = p_a2 - p_a1
    proj_len_ratio_a2 = torch.sum(a2_vector * shot_vector_a2, dim=-1) / (torch.sum(shot_vector_a2**2, dim=-1) + 1e-6)
    is_between_a2 = (proj_len_ratio_a2 > 0) & (proj_len_ratio_a2 < 1)
    dist_perp_sq_a2 = torch.sum((a2_vector - proj_len_ratio_a2.unsqueeze(-1) * shot_vector_a2)**2, dim=-1)
    proximity_factor_a2 = torch.exp(-torch.linalg.norm(a2_vector, dim=-1).pow(2) / (2 * (2 * h_params['agent_radius'])**2))
    line_block_factor = is_between_a2.float() * torch.exp(-dist_perp_sq_a2 / (2 * (0.5 * h_params['agent_radius'])**2))
    line_penalty = h_params['k_a2_shot_line_penalty'] * line_block_factor * proximity_factor_a2
    
    v_closest_def_repel = all_vel[batch_indices, n_attackers + closest_def_indices]
    a1_to_def_vec_repel = p_closest_def - p_a1
    repulsion_direction = a1_to_def_vec_repel / (torch.linalg.norm(a1_to_def_vec_repel, dim=-1, keepdim=True) + 1e-6)
    repulsion_speed = torch.sum(v_closest_def_repel * repulsion_direction, dim=-1)
    dist_a2_to_closest_def = torch.linalg.norm(p_a2 - p_closest_def, dim=-1)
    is_a2_responsible = dist_a2_to_closest_def < h_params['repulsion_proximity_threshold']
    repulsion_reward = h_params['k_repulsion_reward'] * torch.clamp(repulsion_speed, min=0.0) * is_a2_responsible.float()
    dist_a2_to_key_def = torch.linalg.norm(p_a2 - p_closest_def, dim=-1)
    # 使用高斯函数，A2离这个关键防守者越近，奖励越高
    interference_reward = h_params['k_a2_interference_reward'] * torch.exp(-dist_a2_to_key_def.pow(2) / (2 * h_params['screen_pos_sigma']**2))
    dense_reward[:, 1] += screen_reward - line_penalty + repulsion_reward + interference_reward

    # 4.3 防守者奖励/惩罚
    def_pos = all_pos[:, n_attackers:]
    def_vel = all_vel[:, n_attackers:]
    
    in_defensive_half = def_pos[..., 1] > 0
    overextend_depth = torch.where(def_pos[..., 1] < 0, torch.abs(def_pos[..., 1]) + 1, torch.zeros_like(def_pos[..., 1]))
    overextend_penalty = -h_params['k_overextend_penalty'] * overextend_depth
    
    a1_to_basket_vec = basket_pos.unsqueeze(1) - a1_pos.unsqueeze(1)
    a1_to_basket_unit_vec = a1_to_basket_vec / (torch.linalg.norm(a1_to_basket_vec, dim=-1, keepdim=True) + 1e-6)
    ideal_pos = a1_pos.unsqueeze(1) + h_params['def_pos_offset'] * a1_to_basket_unit_vec
    dist_to_ideal = torch.linalg.norm(def_pos - ideal_pos, dim=-1)
    base_positioning_reward = h_params['k_positioning'] * torch.exp(-dist_to_ideal.pow(2) / (2 * h_params['def_pos_sigma']**2))
    d_to_a1_vec = def_pos - a1_pos.unsqueeze(1)
    proj_dot_product = torch.sum(d_to_a1_vec * a1_to_basket_unit_vec, dim=-1)
    soft_gate_factor_def = torch.sigmoid(5.0 * proj_dot_product)
    final_positioning_reward = base_positioning_reward * soft_gate_factor_def * in_defensive_half.float()
    
    a1_in_defensive_half = a1_pos[:, 1] > 0
    is_guarding_context = in_defensive_half & a1_in_defensive_half.unsqueeze(1)
    dist_to_a1 = torch.linalg.norm(def_pos - a1_pos.unsqueeze(1), dim=-1)
    is_close_enough_to_guard = dist_to_a1 < h_params['def_guard_threshold']
    apply_proximity_mask = is_guarding_context & is_close_enough_to_guard
    vec_a1_to_basket = spot_center_pos.unsqueeze(1) - a1_pos.unsqueeze(1)
    unit_vec_a1_to_spot = vec_a1_to_basket / (torch.linalg.norm(vec_a1_to_basket, dim=-1, keepdim=True) + 1e-6)
    vel_a1 = a1_vel
    radial_vel_towards_spot = torch.sum(vel_a1.unsqueeze(1) * unit_vec_a1_to_spot, dim=-1)
    reward_from_a1_movement = -torch.clamp(radial_vel_towards_spot, max=0.0)
    final_spot_control_reward = h_params['k_spot_control_reward'] * reward_from_a1_movement * apply_proximity_mask.float()
    
    dist_d_to_spot_all = torch.linalg.norm(def_pos - spot_center_pos.unsqueeze(1), dim=-1)
    def_gaussian_reward = h_params['k_def_gaussian_spot'] * torch.exp(- (dist_d_to_spot_all**2) / (2 * h_params['def_gaussian_spot_sigma']**2))
    def_gaussian_reward *= in_defensive_half.float()
    
    total_def_reward = overextend_penalty + final_positioning_reward + final_spot_control_reward + def_gaussian_reward
    dense_reward[:, n_attackers:] += total_def_reward

    # --- 5. 时间紧迫性惩罚/奖励 ---
    elapsed_time = h_params['t_limit'] - t_remaining.squeeze(-1)
    apply_mask = elapsed_time > h_params['time_penalty_grace_period']
    if torch.any(apply_mask):
        time_factor = (elapsed_time - h_params['time_penalty_grace_period'])**2
        
        # 进攻方惩罚
        time_penalty_attackers = h_params['k_attacker_time_penalty'] * time_factor
        a1_in_spot = (dist_a1_to_spot <= h_params['R_spot']) & (a1_pos[:, 1] > 0)
        final_penalty_mask = apply_mask & ~a1_in_spot
        # 应用到所有进攻方
        dense_reward -= time_penalty_attackers.unsqueeze(1) * a1_mask * final_penalty_mask.unsqueeze(1)
        
        # 防守方奖励
        time_bonus_defenders = h_params['k_defender_time_bonus'] * time_factor
        # 应用到所有防守方
        dense_reward += time_bonus_defenders.unsqueeze(1) * defender_mask * apply_mask.unsqueeze(1)

    return dense_reward, terminal_rewards, dones_out, curr_still_counter, wall_collision_counters, defender_over_midline_counter, attacker_win_this_step, reason_code