# 导入必要的库
from traceback import print_tb
from sympy import false
import torch
from typing import Dict, Tuple
from vmas import render_interactively
from vmas.simulator.core import World, Agent, Landmark, Sphere, Line
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
import matplotlib
matplotlib.use('Agg') # 对于非交互式绘图至关重要
import matplotlib.pyplot as plt
import io
import pyglet
import numpy as np

from vmas.scenarios.layup_jit import calculate_rewards_and_dones_jit
torch.set_float32_matmul_precision('high')

import time
from functools import partial, update_wrapper

# class timer:
#     """
#     一个通用的、用于计算函数或方法平均运行时间的类装饰器。
#     它实现了描述器协议，可以正确处理类实例的 `self` 参数。
#     """
#     def __init__(self, func):
#         update_wrapper(self, func)
#         self.func = func
#         self.run_times = []
#         self.last_print_time = time.perf_counter()

#     def __get__(self, instance, owner):
#         """实现描述器协议，使其能作为方法装饰器。"""
#         if instance is None:
#             # 如果通过类来访问，例如 MyClass.my_method，则返回装饰器实例本身
#             return self
#         # 如果通过实例来访问，例如 my_instance.my_method
#         # 使用 partial 将实例(instance)和 __call__ 方法绑定在一起
#         # 这就模拟了 Python 的绑定方法机制
#         return partial(self.__call__, instance)

#     def __call__(self, *args, **kwargs):
#         # 注意：如果作为方法装饰器，由于 __get__ 的作用，
#         # 这里的第一个参数 `*args[0]` 将会是类的实例 `self`。
        
#         # 1. 执行函数并计时
#         start_time = time.perf_counter()
#         result = self.func(*args, **kwargs)
#         end_time = time.perf_counter()
        
#         # 2. 存储本次运行时间
#         self.run_times.append(end_time - start_time)
        
#         # 3. 检查是否需要打印平均时间
#         current_time = time.perf_counter()
#         if current_time - self.last_print_time >= 1.0:
#             num_runs = len(self.run_times)
#             avg_time = sum(self.run_times) / num_runs
            
#             # 打印信息
#             # print(f"'{self.func.__name__}' {num_runs} {avg_time * 1e6:.3f} μs")
            
#             # 4. 重置状态
#             self.run_times = []
#             self.last_print_time = current_time
            
#         return result

class Scenario(BaseScenario):
    """
    "飞身上篮"简化版2v2投篮强化学习环境 (已优化和修复版本)
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.viewer_zoom = 3.0
        self.viewer_size = [1400,700]
        # ----------------- 超参数设定 (Hyperparameters) -----------------
        # 创建一个字典来存储所有超参数，以便传递给JIT编译的函数
        self.h_params = {}

        # ========== 1. 场地与物理属性 (Arena & Physics) ==========
        self.h_params["W"] = kwargs.get("W", 8.0)
        self.h_params["L"] = kwargs.get("L", 15.0)
        self.spawn_area_depth = kwargs.get("spawn_area_depth", 1.0)
        self.h_params["R_spot"] = kwargs.get("R_spot", 1.5)
        self.h_params["t_limit"] = kwargs.get("t_limit", 15.0)
        self.dt = kwargs.get("dt", 0.1)
        self.h_params["agent_radius"] = kwargs.get("agent_radius", 0.3)
        self.h_params["a_max"] = kwargs.get("a_max", 3.0)
        self.v_max = kwargs.get("v_max", 5.0)
        
        # ========== 2. 终止条件阈值 (Termination Thresholds) ==========
        self.h_params["v_shot_threshold"] = kwargs.get("v_shot_threshold", 0.2)
        self.h_params["a_shot_threshold"] = kwargs.get("a_shot_threshold", 0.6)
        self.h_params["v_foul_threshold"] = kwargs.get("v_foul_threshold", 0.6)
        self.h_params["shot_still_frames"] = kwargs.get("shot_still_frames", 4)

        self.h_params["max_time_over_midline"] = kwargs.get("max_time_over_midline", 15.0)
        self.h_params["win_condition_block_threshold"] = kwargs.get("win_condition_block_threshold", 0.5)
        
        # ========== 3. 稠密奖励/惩罚系数 (Dense Reward/Penalty Coefficients) ==========
        self.dense_reward_factor = kwargs.get("dense_reward_factor", 0.1)
        self.h_params["k_a1_in_spot_reward"] = kwargs.get("k_a1_in_spot_reward", 3.0)
        self.h_params["k_a1_speed_spot_reward"] = kwargs.get("k_a1_speed_spot_reward", 50.0)
        self.h_params["k_velocity_penalty"] = kwargs.get("k_velocity_penalty", 0.01)
        self.h_params["k_a1_ready_to_shoot_reward"] = kwargs.get("k_a1_ready_to_shoot_reward", 60.0)
        self.h_params["k_a1_velocity_stillness_reward"] = kwargs.get("k_a1_velocity_stillness_reward", 10.0)
        self.h_params["velocity_stillness_sigma"] = kwargs.get("velocity_stillness_sigma", 0.2)
        self.h_params["k_a1_action_stillness_reward"] = kwargs.get("k_a1_action_stillness_reward", 20)
        self.h_params["action_stillness_sigma"] = kwargs.get("action_stillness_sigma", 0.2)
        self.h_params["block_gate_k"] = kwargs.get("block_gate_k", 10.0)
        self.h_params['k_a1_separation_reward'] = kwargs.get("k_a1_separation_reward", 40.0)
        self.h_params["k_a1_blocked_penalty"] = kwargs.get("k_a1_blocked_penalty", -70.0)
        self.h_params["k_a1_tangential_reward"] = kwargs.get("k_a1_tangential_reward", 30.0)
        self.h_params["a1_tangential_pressure_sigma"] = kwargs.get("a1_tangential_pressure_sigma", 1.5)
        self.h_params["gaussian_scale"] = kwargs.get("gaussian_scale", 400.0)
        self.h_params["gaussian_sigma"] = kwargs.get("gaussian_sigma", 0.6 * self.h_params["R_spot"])
        self.h_params["low_u_threshold"] = kwargs.get("low_u_threshold", 0.9)
        self.h_params["hesitate_speed_threshold"] = kwargs.get("hesitate_speed_threshold", 1)
        self.h_params["k_hesitation_penalty"] = kwargs.get("k_hesitation_penalty", 10)
        self.h_params["a1_proximity_threshold"] = kwargs.get("a1_proximity_threshold", self.h_params["agent_radius"] * 3.0)
        self.h_params["a1_proximity_penalty_margin"] = kwargs.get("a1_proximity_penalty_margin", 0.01)
        self.h_params["k_a1_proximity_penalty"] = kwargs.get("k_a1_proximity_penalty", 90)
        self.h_params["k_ideal_screen_pos"] = kwargs.get("k_ideal_screen_pos", 80.0)
        self.h_params["k_repulsion_reward"] = kwargs.get("k_repulsion_reward", 1.0)
        self.h_params["repulsion_proximity_threshold"] = kwargs.get("repulsion_proximity_threshold", self.h_params["R_spot"])
        self.h_params["k_a2_shot_line_penalty"] = kwargs.get("k_a2_shot_line_penalty", 10)
        self.h_params["screen_pos_offset"] = kwargs.get("screen_pos_offset", self.h_params["agent_radius"] * 3)
        self.h_params["screen_pos_sigma"] = kwargs.get("screen_pos_sigma", self.h_params["R_spot"])
        self.h_params["k_screen_gate"] = kwargs.get("k_screen_gate", 7.0)
        self.h_params["screen_spacing_gate_k"] = kwargs.get("screen_spacing_gate_k", 7.0)
        self.h_params["k_a2_interference_reward"] = kwargs.get("k_a2_interference_reward", 40.0)
        self.h_params["time_penalty_grace_period"] = kwargs.get("time_penalty_grace_period", 9)
        self.h_params["k_attacker_time_penalty"] = kwargs.get("k_attacker_time_penalty", 1)
        self.h_params["k_defender_time_bonus"] = kwargs.get("k_defender_time_bonus", 0.02)
        self.h_params["k_positioning"] = kwargs.get("k_positioning", 300.0)
        self.h_params["k_spot_control_reward"] = kwargs.get("k_spot_control_reward", 2.0) # 奖励系数，用于奖励成功迫使A1远离投篮点的防守行为
        self.h_params["def_guard_threshold"] = kwargs.get("def_guard_threshold", self.h_params["agent_radius"] * 8.0) # 盯防判定距离
        self.h_params["k_overextend_penalty"] = kwargs.get("k_overextend_penalty", 240.0)
        self.h_params["k_def_gaussian_spot"] = kwargs.get("k_def_gaussian_spot", 50)
        self.h_params["def_gaussian_spot_sigma"] = kwargs.get("def_gaussian_spot_sigma", 1*self.h_params["R_spot"])
        self.h_params["def_pos_offset"] = kwargs.get("def_pos_offset", self.h_params["agent_radius"] * 2.5)
        self.h_params["def_pos_sigma"] = kwargs.get("def_pos_sigma", 1.5)
        self.h_params["oob_penalty"] = kwargs.get("oob_penalty", -3000.0)
        self.h_params["oob_margin"] = kwargs.get("oob_margin", 0.05)
        self.h_params["k_coll_active"] = kwargs.get("k_coll_active", 5.0)
        self.h_params["k_coll_passive"] = kwargs.get("k_coll_passive", 0.1)
        self.h_params["k_u_penalty_general"] = kwargs.get("k_u_penalty_general", 0.001)
        self.h_params["proximity_threshold"] = kwargs.get("proximity_threshold", self.h_params["agent_radius"] * 2.2)
        self.h_params["proximity_penalty_margin"] = kwargs.get("proximity_penalty_margin", 0.15)
        self.h_params["k_proximity_penalty"] = kwargs.get("k_proximity_penalty", 120)
        self.h_params["k_def_proximity_penalty"] = kwargs.get("k_def_proximity_penalty", 120.0)
        self.h_params["proximity_penalty_reduction_in_spot"] = kwargs.get("proximity_penalty_reduction_in_spot", 0.2)
        self.h_params["low_velocity_threshold"] = kwargs.get("low_velocity_threshold", self.h_params['v_foul_threshold'])
        self.h_params["k_push_penalty"] = kwargs.get("k_push_penalty", 500.0)
        self.h_params["k_def_push_penalty"] = kwargs.get("k_def_push_penalty", 500.0)
        self.h_params["stand_still_threshold"] = kwargs.get("stand_still_threshold", self.h_params['v_foul_threshold'])
        self.h_params["k_stand_still_reward"] = kwargs.get("k_stand_still_reward", 20.0)
        self.h_params["charge_drawing_range"] = kwargs.get("charge_drawing_range", self.h_params["agent_radius"] * 6.0)
        self.h_params["k_excess_acceleration_penalty"] = kwargs.get("k_excess_acceleration_penalty", 0.002)
        self.h_params["k_action_jerk_penalty"] = kwargs.get("k_action_jerk_penalty", 0.05)
        
        # ========== 4. 终局奖励/惩罚系数 (Terminal Reward/Penalty Coefficients) ==========
        self.h_params["max_score"] = kwargs.get("max_score", 4000.0)
        self.h_params["shoot_score"] = kwargs.get("shoot_score", 2000.0)
        self.h_params["k_spacing_bonus"] = kwargs.get("k_spacing_bonus", 1000.0)
        self.h_params["k_a2_screen_bonus"] = kwargs.get("k_a2_screen_bonus", 2000.0)
        self.h_params["a2_screen_sigma"] = kwargs.get("a2_screen_sigma", 4 * self.h_params["agent_radius"])
        self.h_params["k_time_bonus"] = kwargs.get("k_time_bonus", 2000.0) # 投篮时间奖励系数
        self.h_params["R_foul"] = kwargs.get("R_foul", 3000.0)
        self.h_params["k_foul_vel_penalty"] = kwargs.get("k_foul_vel_penalty", 200.0)
        self.h_params["attacker_timeout_penalty"] = kwargs.get("attacker_timeout_penalty", 800)
        self.h_params["k_timeout_dist_penalty"] = kwargs.get("k_timeout_dist_penalty", 300.0)
        self.h_params["attacker_timeout_penalty_max"] = kwargs.get("attacker_timeout_penalty_max", 2000.0)
        self.h_params["k_def_block_reward"] = kwargs.get("k_def_block_reward", 800.0)
        self.h_params["k_def_force_reward"] = kwargs.get("k_def_force_reward", 200.0)
        self.h_params["k_def_pos_reward"] = kwargs.get("k_def_pos_reward", 20.0)
        self.h_params["k_def_area_reward"] = kwargs.get("k_def_area_reward", 100.0)
        self.h_params["k_def_shot_penalty"] = kwargs.get("k_def_shot_penalty", 300.0)
        self.h_params["foul_teammate_factor"] = kwargs.get("foul_teammate_factor", 0.1)
        self.h_params["wall_collision_frames"] = kwargs.get("wall_collision_frames", 10.0)
        self.h_params["R_wall_collision_penalty"] = kwargs.get("R_wall_collision_penalty", -9000.0)
        self.h_params["R_midline_foul"] = kwargs.get("R_midline_foul", 7000.0)
        
        # ========== 5. 其他行为控制参数 (Other Behavior Control) ==========
        self.start_delay_frames = kwargs.get("start_delay_frames", 10)
        self.h_params["def_proximity_threshold"] = kwargs.get("def_proximity_threshold", 1.2)
        self.h_params["block_sigma"] = kwargs.get("block_sigma", self.h_params["agent_radius"] * 1.5)
        
        # ----------------- 环境构建 (World Setup) -----------------
        self.max_steps = int(self.h_params["t_limit"] / self.dt)
        self.n_agents = 4
        self.n_attackers = 2
        self.n_defenders = 2

        world = World(batch_dim, device, dt=self.dt, substeps=4,
                      x_semidim=self.h_params["W"] / 2, y_semidim=self.h_params["L"] / 2)

        for i in range(self.n_agents):
            is_attacker = i < self.n_attackers
            team_name = "attacker" if is_attacker else "defender"
            agent_id = i + 1 if is_attacker else i - self.n_attackers + 1
            agent = Agent(
                name=f"{team_name}_{agent_id}",
                collide=True,
                movable=True,
                rotatable=False,
                u_range=self.v_max,
                drag=0.01,
                shape=Sphere(radius=self.h_params["agent_radius"]),
                dynamics=Holonomic(),
                render_action=True,
                color=Color.RED if is_attacker and agent_id == 1 else Color.BLUE if not is_attacker else Color.PINK
            )
            agent.is_attacker = is_attacker
            agent.controller = VelocityController(agent, world, [6,0,0.01], "parallel")
            world.add_agent(agent)

        self.attackers = world.agents[:self.n_attackers]
        self.defenders = world.agents[self.n_attackers:]
        self.a1 = self.attackers[0]
        self.a2 = self.attackers[1]

        self.basket = Landmark(name="basket", collide=False, shape=Sphere(radius=0.1), color=Color.ORANGE)
        self.spot_center = Landmark(name="spot_center", collide=False, shape=Sphere(radius=0.05), color=Color.GREEN)
        self.shooting_area_vis = Landmark(name="shooting_area_vis", collide=False, shape=Sphere(radius=self.h_params["R_spot"]), color=Color.LIGHT_GREEN)
        center_line = Landmark(name="center_line", collide=False, shape=Line(length=self.h_params["W"]), color=Color.GRAY)
        world.add_landmark(center_line)
        world.add_landmark(self.basket)
        world.add_landmark(self.spot_center)
        world.add_landmark(self.shooting_area_vis)

        # 初始化内部状态变量
        self.t_remaining = torch.zeros(batch_dim, 1, device=device)
        self.step_dense_rewards = torch.zeros(batch_dim, self.n_agents, device=device) # 用于存储当前步的稠密奖励
        self.terminal_rewards = torch.zeros(batch_dim, self.n_agents, device=device)   # 用于存储终局奖励
        self.dones = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.p_vels = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.raw_actions = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.delay_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.a1_still_frames_counter = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self.wall_collision_counters = torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32)
        self.defender_over_midline_counter = torch.zeros((batch_dim, self.n_defenders), device=device, dtype=torch.int32)
        self.win_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.dones_this_step = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.requested_accelerations = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.p_raw_actions = torch.zeros((batch_dim, self.n_agents, 2), device=device)
        self.termination_reason_code = torch.zeros(batch_dim, device=device, dtype=torch.int32)


        # self.jitted_reward_calculator = torch.compile(calculate_rewards_and_dones_jit, mode="max-autotune")
        self.jitted_reward_calculator = torch.jit.script(calculate_rewards_and_dones_jit)

        self.reward_hist = {}

        self.plot_artists = []
        for i, agent in enumerate(world.agents):
            fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
            fig.tight_layout(pad=1)
            line, = ax.plot([], [], 'r-') 
            ax.set_title(f"Agent {agent.name}", fontsize=6)
            artist_dict = {'fig': fig, 'ax': ax, 'line': line}
            self.plot_artists.append(artist_dict)

        return world

    # @timer
    def reset_world_at(self, env_index: int | None = None):
        """
        根据指定的生成规则，使用高效的并行计算和局部抖动网格重置环境。
        - a1: 位置固定。
        - a2: 在靠近中线的己方条带内随机生成。
        - d1, d2: 在靠近中线的己方条带内使用抖动网格生成，以避免碰撞。
        此方法通过构造保证了所有智能体无初始碰撞且满足边界条件。
        """
        if env_index is None:
            batch_range = slice(None)
            batch_dim = self.world.batch_dim
            self.reward_hist.clear()
        else:
            batch_range = env_index
            batch_dim = 1 if isinstance(env_index, int) else len(env_index)
            if isinstance(env_index, int):
                if env_index in self.reward_hist:
                    del self.reward_hist[env_index]
            else: # env_index is a list or tensor
                for idx in env_index:
                    if idx in self.reward_hist:
                        del self.reward_hist[idx]
        
        # 重置时间和内部状态
        self.t_remaining[batch_range] = self.h_params["t_limit"]
        self.terminal_rewards[batch_range] = 0.0
        self.p_vels[batch_range] = 0.0
        self.delay_counter[batch_range] = self.start_delay_frames
        self.a1_still_frames_counter[batch_range] = 0
        self.wall_collision_counters[batch_range] = 0
        self.defender_over_midline_counter[batch_range] = 0
        self.dones[batch_range] = 0
        self.p_raw_actions[batch_range] = 0.0
        self.termination_reason_code[batch_range] = 0

        # 随机化篮筐和投篮点位置
        basket_pos = torch.zeros(batch_dim, 2, device=self.world.device)
        basket_pos[:, 1] = self.h_params["L"] / 2 - 0.6
        self.basket.set_pos(basket_pos, batch_index=env_index)

        spot_x = (torch.rand(batch_dim, 1, device=self.world.device) - 0.5) * (self.h_params["W"]-self.h_params["R_spot"])
        spot_y = torch.rand(batch_dim, 1, device=self.world.device) * (self.h_params["L"] / 4) + (self.h_params["R_spot"]/2)
        spot_pos = torch.cat([spot_x, spot_y], dim=1)
        self.spot_center.set_pos(spot_pos, batch_index=env_index)
        self.shooting_area_vis.set_pos(spot_pos, batch_index=env_index)

        # ---------- 高效并行化智能体放置 ----------
        # 1. 获取参数
        W, L = self.h_params["W"], self.h_params["L"]
        agent_radius = self.h_params["agent_radius"]
        spawn_area_depth = self.spawn_area_depth
        n_defenders = self.n_defenders
        device = self.world.device

        # --- 攻击方1 (a1): 固定位置 ---
        # 固定在场地左下角，并留出自身半径的边界
        pos_a1_x = -W / 2 + agent_radius * 2
        pos_a1_y = -L / 2 + agent_radius * 2
        pos_a1 = torch.tensor([[pos_a1_x, pos_a1_y]], device=device, dtype=torch.float32).expand(batch_dim, -1)

        # --- 攻击方2 (a2): 在己方条带内随机生成 ---
        # 生成区域: X轴在[-W/2, W/2]内，Y轴在[-spawn_area_depth, 0]内
        # 为避免在边缘生成，所有计算均考虑agent_radius的边界
        valid_width = W - 2 * agent_radius
        valid_depth = spawn_area_depth - agent_radius # 避免生成在y=0中线上

        pos_a2_x = (torch.rand(batch_dim, 1, device=device) - 0.5) * valid_width
        # 在Y轴 [-spawn_area_depth, -agent_radius] 区间内生成
        pos_a2_y = -agent_radius - torch.rand(batch_dim, 1, device=device) * valid_depth
        pos_a2 = torch.cat([pos_a2_x, pos_a2_y], dim=1)

        # --- 防守方 (d1, d2): 在己方条带内使用局部抖动网格 ---
        # 生成区域: X轴在[-W/2, W/2]内，Y轴在[0, spawn_area_depth]内
        # 使用1x2的网格放置2个防守方，以从结构上避免碰撞
        
        # 定义网格单元尺寸
        def_cell_w = valid_width / n_defenders
        
        # 计算抖动范围
        max_jitter_x = max(0.0, (def_cell_w / 2) - agent_radius)
        max_jitter_y = max(0.0, valid_depth / 2)

        # 生成随机抖动值
        def_jitter = (torch.rand(batch_dim, n_defenders, 2, device=device) - 0.5)
        def_jitter[:, :, 0] *= 2 * max_jitter_x
        def_jitter[:, :, 1] *= 2 * max_jitter_y

        # 计算网格单元中心点（基础位置），并随机分配智能体到单元
        def_indices = torch.rand(batch_dim, n_defenders, device=device).argsort(dim=1)
        def_base_x = -valid_width/2 + def_cell_w/2 + def_indices * def_cell_w
        def_base_y = torch.full_like(def_base_x, agent_radius + valid_depth / 2)
        def_base_pos = torch.stack([def_base_x, def_base_y], dim=-1)

        # 计算防守方最终位置
        pos_def = def_base_pos + def_jitter

        # --- 组合所有智能体位置 ---
        agent_positions = torch.cat([pos_a1.unsqueeze(1), pos_a2.unsqueeze(1), pos_def], dim=1)
        
        # 设置智能体状态
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(agent_positions[:, i, :], batch_index=env_index)
            agent.set_vel(torch.zeros(batch_dim, 2, device=self.world.device), batch_index=env_index)

    # @timer
    def process_action(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        # 保存模型输出的原始动作（期望速度）
        self.raw_actions[:, agent_idx, :] = agent.action.u.clone()
        # 在开局延迟期内，A1不能移动
        if agent == self.a1:
            is_delayed = self.delay_counter > 0
            agent.action.u[is_delayed] = 0.0
        # 动作死区，忽略过小的动作输入
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.1] = 0.0
        # 将期望速度限制在最大速度范围内
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
        # 根据物理限制（最大加速度）计算本帧可达到的速度
        requested_a = (agent.action.u - agent.state.vel) / self.world.dt
        self.requested_accelerations[:, agent_idx, :] = requested_a
        achievable_a = TorchUtils.clamp_with_norm(requested_a, self.h_params["a_max"])
        agent.action.u = agent.state.vel + achievable_a * self.world.dt
        # 调用底层控制器执行最终计算出的速度
        agent.controller.process_force()

    # @timer
    def pre_step(self):
        """
        在每个物理步长开始前执行。
        这是执行JIT函数的最佳位置，因为它在所有智能体动作处理后、物理引擎步进前运行。
        """
        self.win_this_step.zero_()
        self.t_remaining -= self.world.dt
        self.delay_counter = torch.clamp(self.delay_counter - 1, min=0)

        # 1. 收集所有智能体的状态张量
        self.all_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        self.all_vel = torch.stack([a.state.vel for a in self.world.agents], dim=1)
        
        # 2. 预计算所有智能体间的交互信息，以供JIT函数使用
        self.pos_diffs = self.all_pos.unsqueeze(2) - self.all_pos.unsqueeze(1)
        self.dist_matrix = torch.linalg.norm(self.pos_diffs, dim=-1)
        self.collision_matrix = self.dist_matrix < (self.h_params["agent_radius"] * 2)
        self.collision_matrix.diagonal(dim1=-2, dim2=-1).fill_(False)

        self.vel_diffs = self.all_vel.unsqueeze(2) - self.all_vel.unsqueeze(1)
        self.vel_diffs_norm = torch.linalg.norm(self.vel_diffs, dim=-1)

        # 3. 更新撞墙计数器
        wall_x = self.world.x_semidim * 0.999
        wall_y = self.world.y_semidim * 0.999
        is_pushing_wall_x = (self.all_pos[..., 0] > wall_x) | (self.all_pos[..., 0] < -wall_x)
        is_pushing_wall_y = (self.all_pos[..., 1] > wall_y) | (self.all_pos[..., 1] < -wall_y)
        is_pushing_wall = is_pushing_wall_x | is_pushing_wall_y
        
        wall_counters_clone = self.wall_collision_counters.clone()
        wall_counters_clone[is_pushing_wall] += 1
        wall_counters_clone[~is_pushing_wall] = 0 # 没有推墙则清零，实现“连续”检测
        self.wall_collision_counters.copy_(wall_counters_clone)

        # 4. 调用核心JIT函数进行计算
        dense_rewards, terminal_rewards, dones, a1_still_frames_counter, wall_collision_counters, defender_over_midline_counter, win_this_step, updated_reason_code = \
            self.jitted_reward_calculator(
                self.h_params,
                self.all_pos,
                self.all_vel,
                self.p_vels,
                self.p_raw_actions,
                self.raw_actions,
                self.basket.state.pos,
                self.spot_center.state.pos,
                self.t_remaining,
                self.a1_still_frames_counter.to(torch.int32), # 确保传入JIT的类型正确
                self.wall_collision_counters.to(torch.int32),
                self.defender_over_midline_counter.to(torch.int32),
                self.termination_reason_code.to(torch.int32),
                self.dones,
                self.dist_matrix,
                self.collision_matrix,
                self.vel_diffs_norm,
                self.requested_accelerations,
            )

        # 5. 根据JIT函数的输出更新场景状态
        self.step_dense_rewards = dense_rewards
        self.terminal_rewards = terminal_rewards
        self.dones = dones
        self.a1_still_frames_counter = a1_still_frames_counter.to(torch.int32)
        self.wall_collision_counters = wall_collision_counters.to(torch.int32)
        self.defender_over_midline_counter = defender_over_midline_counter.to(torch.int32)
        self.win_this_step = win_this_step
        self.termination_reason_code = updated_reason_code.to(torch.int32)

        # # 可以在这里处理JIT函数无法执行的操作，比如打印
        # if torch.any(self.win_this_step):
        #     print(f"got {torch.sum(self.win_this_step).item()} wins in this step")
        
        self.dones_this_step.copy_(self.dones)

    # @timer
    def post_step(self):
        """
        在物理步长结束后执行，用于记录状态以备下一帧使用。
        """
        # 记录当前帧的速度，作为下一帧的“上一帧速度”
        self.p_vels.copy_(self.all_vel)
        self.p_raw_actions.copy_(self.raw_actions)

        # 对物理出界的智能体，将其速度强制置零
        for agent in self.world.agents:
            pos = agent.state.pos
            is_hard_oob = (torch.abs(pos[:, 0]) > (0.999 * self.h_params['W'] / 2)) | (torch.abs(pos[:, 1]) > (0.999 * self.h_params['L'] / 2))
            agent.state.vel[is_hard_oob] = 0.0

    def info(self, agent: Agent):
        # 获取当前智能体的索引
        agent_idx = self.world.agents.index(agent)
        
        # 从预先计算好的奖励张量中，根据索引提取对应的值
        # .clone() 和 .unsqueeze(-1) 是为了保证格式正确
        dense_reward = self.dense_reward_factor * self.step_dense_rewards[:, agent_idx].clone().unsqueeze(-1)
        terminal_reward = self.terminal_rewards[:, agent_idx].clone().unsqueeze(-1)

        return {
            # 原有的信息
            "win_in_step": self.win_this_step.clone().float().unsqueeze(-1),
            "termination_reason": self.termination_reason_code.clone().float().unsqueeze(-1),
            
            # 新增的奖励信息
            "dense_reward": dense_reward,
            "terminal_reward": terminal_reward,
        }

    def done(self):
        # 直接返回在pre_step中由JIT函数计算好的dones标志
        return self.dones

    def reward(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        
        # 核心计算已在pre_step中完成。这里只负责组合奖励并返回。
        # 最终奖励 = 稠密奖励 * 系数 + 终局奖励
        rew = self.dense_reward_factor * self.step_dense_rewards[:, agent_idx] + self.terminal_rewards[:, agent_idx]

        # 在开局延迟期内，A1的奖励为0
        if agent == self.a1:
            is_delayed = self.delay_counter > 0
            rew = torch.where(is_delayed, 0.0, rew)
        return rew

    # @timer
    def observation(self, agent: Agent):
        agent_idx = self.world.agents.index(agent)
        is_attacker = agent_idx < self.n_attackers

        # 根据角色定义观测信息
        if is_attacker:
            teammate = self.attackers[1 - agent_idx]
            opp1, opp2 = self.defenders[0], self.defenders[1]
            # 进攻方关注投篮点
            key_info_rel = self.spot_center.state.pos - agent.state.pos
        else:
            teammate = self.defenders[1 - (agent_idx - self.n_attackers)]
            opp1, opp2 = self.attackers[0], self.attackers[1]
            # 防守方关注A1到篮筐的路径
            key_info_rel = self.basket.state.pos - self.a1.state.pos

        # 将所有信息拼接成一个观测向量
        obs = torch.cat([
            agent.state.pos,                              # 自身绝对位置
            agent.state.vel,                              # 自身绝对速度
            teammate.state.pos - agent.state.pos,         # 队友相对位置
            teammate.state.vel - agent.state.vel,         # 队友相对速度
            opp1.state.pos - agent.state.pos,             # 对手1相对位置
            opp1.state.vel - agent.state.vel,             # 对手1相对速度
            opp2.state.pos - agent.state.pos,             # 对手2相对位置
            opp2.state.vel - agent.state.vel,             # 对手2相对速度
            key_info_rel,                                 # 关键目标信息 (对进攻/防守方不同)
            self.t_remaining / self.h_params["t_limit"],  # 归一化的剩余时间
        ], dim=-1)

        return obs
    
    def extra_render(self, env_index: int):
        # 此部分用于在渲染窗口中额外绘制调试信息（如奖励曲线图）
        geoms = []
        from vmas.simulator.rendering import Geom
        import io
        
        
        class SpriteGeom(Geom):
            def __init__(self, image, x, y, target_width, target_height):
                super().__init__()
                texture = image.get_texture()
                flipped_texture = texture.get_transform(flip_y=True)
                self.sprite = pyglet.sprite.Sprite(img=flipped_texture, x=x, y=y)
                if self.sprite.width > 0: self.sprite.scale_x = target_width / self.sprite.width
                if self.sprite.height > 0: self.sprite.scale_y = target_height / self.sprite.height
                self.sprite.blend_src = pyglet.gl.GL_SRC_ALPHA
                self.sprite.blend_dest = pyglet.gl.GL_ONE_MINUS_SRC_ALPHA
            def render1(self):
                self.sprite.draw()

        plot_width = 10
        plot_height = 6
        pose_list = [(-14, 0), (4, 0), (-14, -6), (4, -6)] 
        # 遍历每个智能体，更新其历史并绘图
        for i, agent in enumerate(self.world.agents):
            # 1. 计算当前步的奖励
            rew_tensor = self.reward(agent)
            # 2. 将当前环境的奖励值(标量)追加到历史记录中
            if env_index not in self.reward_hist:
                self.reward_hist[env_index] = {}
            if i not in self.reward_hist[env_index]:
                self.reward_hist[env_index][i] = []
            self.reward_hist[env_index][i].append(rew_tensor[env_index].item())

            # 3. 准备绘图
            history_list = self.reward_hist[env_index][i]
            artists = self.plot_artists[i]
            fig, ax, line = artists['fig'], artists['ax'], artists['line']
            
            x_data = range(len(history_list))
            line.set_data(x_data, history_list)
            ax.relim()
            ax.autoscale_view(tight=True)
            
            # 4. 将matplotlib图像转换为Pyglet可渲染对象
            with io.BytesIO() as buf:
                fig.canvas.draw()
                image_data = fig.canvas.buffer_rgba().tobytes()
                plot_image = pyglet.image.ImageData(
                    fig.canvas.get_width_height()[0],
                    fig.canvas.get_width_height()[1],
                    'RGBA',
                    image_data
                )
                if i < len(pose_list):
                    x, y = pose_list[i]
                    img_geom = SpriteGeom(plot_image, x, y + 6, plot_width, plot_height)
                    geoms.append(img_geom)
        return geoms

if __name__ == "__main__":
    # 使用此脚本可以交互式地运行和测试环境
    render_interactively(
        __file__,
        control_two_agents=True, # 允许手动控制两个智能体进行测试
    )