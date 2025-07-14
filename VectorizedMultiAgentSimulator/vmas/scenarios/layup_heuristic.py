import torch
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.utils import TorchUtils

class HeuristicPolicy_BoundaryAwarePlanner(BaseHeuristicPolicy):
    """
    最终版: 边界感知的几何路径规划器
    - 修复了绕行点可能在场地外导致卡死的问题。
    - 使用代价函数来选择最优绕行点，同时考虑路径效率和出界惩罚。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PD控制器参数
        self.kp = 5.0
        self.kd = 4.0
        
        # 避障参数
        self.safe_distance = 0.3 * 2 + 0.3
        
        # (HIGHLIGHT) 获取并存储场地边界信息
        self.world_w, self.world_l = (8,15)
        self.x_bound = self.world_w / 2
        self.y_bound = self.world_l / 2
        
        # (HIGHLIGHT) 出界惩罚的权重，设置得非常高
        self.out_of_bounds_penalty_weight = 1000.0

    def compute_action(self, observation: torch.Tensor, u_range: torch.Tensor) -> torch.Tensor:
        is_batched = True
        if observation.dim() == 1:
            is_batched = False
            observation = observation.unsqueeze(0)

        # --- 1. 解析状态 ---
        self_pos = observation[:, 0:2]
        self_vel = observation[:, 2:4]
        obstacles_pos = [
            self_pos + observation[:, 4:6],
            self_pos + observation[:, 8:10],
            self_pos + observation[:, 12:14],
        ]
        vec_to_goal = observation[:, 16:18]
        goal_pos = self_pos + vec_to_goal

        # --- 2. 几何路径规划 ---
        sub_goal_pos = goal_pos.clone()
        
        min_dist_to_collision = torch.full((observation.shape[0],), float('inf'), device=observation.device)
        closest_obstacle_pos = torch.zeros_like(self_pos)
        path_is_blocked = torch.zeros((observation.shape[0],), dtype=torch.bool, device=observation.device)
        direction_to_goal = vec_to_goal / (torch.linalg.norm(vec_to_goal, dim=1, keepdim=True) + 1e-6)
        
        for obs_pos in obstacles_pos:
            vec_to_obs = obs_pos - self_pos
            proj_len = torch.sum(vec_to_obs * direction_to_goal, dim=1)
            is_in_front = proj_len > 0
            dist_perp_sq = torch.sum(vec_to_obs**2, dim=1) - proj_len**2
            is_colliding = (dist_perp_sq < self.safe_distance**2) & is_in_front & (proj_len < torch.linalg.norm(vec_to_goal, dim=1))

            if torch.any(is_colliding):
                path_is_blocked |= is_colliding
                dist_to_obs = torch.linalg.norm(vec_to_obs, dim=1)
                is_closer = is_colliding & (dist_to_obs < min_dist_to_collision)
                closest_obstacle_pos = torch.where(is_closer.unsqueeze(1), obs_pos, closest_obstacle_pos)
                min_dist_to_collision = torch.where(is_closer, dist_to_obs, min_dist_to_collision)

        if torch.any(path_is_blocked):
            blocked_indices = path_is_blocked
            vec_to_closest_obs = closest_obstacle_pos[blocked_indices] - self_pos[blocked_indices]
            dist_to_obs = torch.linalg.norm(vec_to_closest_obs, dim=1, keepdim=True)
            angle_alpha = torch.acos(torch.clamp(self.safe_distance / dist_to_obs, -1.0, 1.0))
            
            cos_a, sin_a = torch.cos(angle_alpha), torch.sin(angle_alpha)
            rot_mat1 = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1).view(-1, 2, 2)
            escape_vec1 = torch.bmm(rot_mat1, vec_to_closest_obs.unsqueeze(-1)).squeeze(-1)
            escape_point1 = self_pos[blocked_indices] + escape_vec1
            
            rot_mat2 = torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(-1, 2, 2)
            escape_vec2 = torch.bmm(rot_mat2, vec_to_closest_obs.unsqueeze(-1)).squeeze(-1)
            escape_point2 = self_pos[blocked_indices] + escape_vec2

            # --- (HIGHLIGHT) 使用代价函数选择最优绕行点 ---
            
            # a. 计算路径代价
            path_cost1 = torch.linalg.norm(escape_point1 - self_pos[blocked_indices], dim=1) + torch.linalg.norm(goal_pos[blocked_indices] - escape_point1, dim=1)
            path_cost2 = torch.linalg.norm(escape_point2 - self_pos[blocked_indices], dim=1) + torch.linalg.norm(goal_pos[blocked_indices] - escape_point2, dim=1)

            # b. 计算出界惩罚
            def out_of_bounds_cost(p):
                # 超出边界的距离，不出界则为0
                oob_x = torch.clamp(torch.abs(p[:, 0]) - self.x_bound, min=0)
                oob_y = torch.clamp(torch.abs(p[:, 1]) - self.y_bound, min=0)
                return (oob_x + oob_y) * self.out_of_bounds_penalty_weight
            
            oob_cost1 = out_of_bounds_cost(escape_point1)
            oob_cost2 = out_of_bounds_cost(escape_point2)

            # c. 计算总代价
            total_cost1 = path_cost1 + oob_cost1
            total_cost2 = path_cost2 + oob_cost2
            
            chosen_escape_point = torch.where((total_cost1 < total_cost2).unsqueeze(1), escape_point1, escape_point2)
            sub_goal_pos[blocked_indices] = chosen_escape_point

        # --- 3. & 4. PD控制器与执行 (逻辑不变) ---
        pos_error_to_subgoal = sub_goal_pos - self_pos
        vel_error = self_vel
        a_target = self.kp * pos_error_to_subgoal - self.kd * vel_error
        
        a_max = u_range
        a_target_norm = torch.linalg.norm(a_target, dim=1, keepdim=True)
        a_target = torch.where(a_target_norm > a_max, a_target / a_target_norm * a_max, a_target)

        dt = 0.1
        desired_velocity = self_vel + a_target * dt
        action = TorchUtils.clamp_with_norm(desired_velocity, u_range)
        
        if not is_batched:
            action = action.squeeze(0)
            
        return action