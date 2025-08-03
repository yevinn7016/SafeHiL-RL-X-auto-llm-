#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:02:04 2023

@author: oscar
"""

import os
import gym  # classic gym for SMARTS 0.6.1
import sys
import yaml
import time
import torch
import warnings
import statistics
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from collections import deque
import matplotlib.pyplot as plt


sys.path.append('/home/oscar/Dropbox/SMARTS')
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv  # SMARTS 0.6.1 classic import
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap, Waypoints


from drl_agent import DRL
from keyboard import HumanKeyboardAgent
from utils_ import soft_update, hard_update
from authority_allocation import Arbitrator
import sys
sys.path.append('./SMARTS')
from memory_module import MemoryModule
import sys
import os
from datetime import datetime

# 로그 디렉토리 생성
os.makedirs("logs", exist_ok=True)

# 현재 시간 기반 로그 파일 이름
log_filename = datetime.now().strftime("logs/log_%Y%m%d_%H%M%S.txt")

# 표준 출력 로그 저장
sys.stdout = open(log_filename, "w")
sys.stderr = sys.stdout  # 에러 출력도 함께 저장하고 싶다면

def plot_animation_figure(epoc):
    plt.figure()
    plt.clf()
    
    plt.subplot(1, 1, 1)
    plt.title(env_name + ' ' + name + ' Save Epoc:' + str(epoc) +
              ' Alpha:' + str(agent.alpha))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list)
    plt.plot(reward_mean_list)

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.tight_layout()
    plt.show()

def vec_2d(v) -> np.ndarray:
    """Converts a higher order vector to a 2D vector."""

    assert len(v) >= 2
    return np.array(v[:2])

def signedDistToLine(point, line_point, line_dir_vec) -> float:
    p = vec_2d(point)
    p1 = line_point
    p2 = line_point + line_dir_vec

    u = abs(
        line_dir_vec[1] * p[0] - line_dir_vec[0] * p[1] + p2[0] * p1[1] - p2[1] * p1[0]
    )
    d = u / np.linalg.norm(line_dir_vec)

    line_normal = np.array([-line_dir_vec[1], line_dir_vec[0]])
    _sign = np.sign(np.dot(p - p1, line_normal))
    return d * _sign

def evaluate(env_eval, agent, eval_episodes=10, epoch=0):
    ep = 0
    success = int(0)
    avg_reward_list = []
    lane_center = [-3.2, 0, 3.2]

    v_list_avg = []
    offset_list_avg = []
    dist_list_avg = []

    while ep < eval_episodes:
        obs = env_eval.reset()
        obs = obs[AGENT_ID]
        s = observation_adapter(obs)
        done = False
        reward_total = 0.0
        frame_skip = 5

        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))

        initial_pos = obs.ego_vehicle_state.position[:2]
        pos_list = deque(maxlen=5)
        pos_list.append(initial_pos)

        df = pd.DataFrame([])
        s_list = []
        l_list = []
        offset_list = []
        v_list = []
        steer_list = []

        for t in count():
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break

            # ===== 선택 및 수행 =====
            a, llm_action = agent.choose_action(np.array(s), action_list[-1], evaluate=True)
            action = action_adapter(a)

            # ===== 안전 마스크 =====
            ego_state = obs.ego_vehicle_state
            lane_id = ego_state.lane_index
            if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and action[0] > 0.0:
                action = list(action)
                action[0] = 0.0
                action = tuple(action)

            action = {AGENT_ID: action}
            next_obs, reward, done, info = env_eval.step(action)

            obs = next_obs[AGENT_ID]
            s_ = observation_adapter(next_obs[AGENT_ID])
            if isinstance(done, dict):
                done = done.get(AGENT_ID, False)
            curr_pos = next_obs[AGENT_ID].ego_vehicle_state.position[:2]
            engage = int(0)

            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')

            if mode == 'evaluation':
                r = reward_adapter(next_obs[AGENT_ID], pos_list, a, engage, done=done, llm_action=None)
            else:
                r = reward_adapter(next_obs[AGENT_ID], pos_list, a, engage, done=done, llm_action=llm_action)
            pos_list.append(next_obs[AGENT_ID].ego_vehicle_state.position[:2])
            action_list.append(a)
            s = s_

            # ===== 여기서 lateral offset 계산 변경 =====
            ego_pos = ego_state.position[:2]
            if lane_id < len(obs.waypoint_paths):
                path = obs.waypoint_paths[lane_id]
                if len(path) >= 2:
                    ref_start = path[0].pos[:2]
                    ref_end = path[1].pos[:2]
                    ref_dir = ref_end - ref_start
                    normal = np.array([-ref_dir[1], ref_dir[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-6)
                    lateral_offset = np.dot(ego_pos - ref_start, normal)
                else:
                    lateral_offset = 0.0
            else:
                lateral_offset = 0.0

            # ===== 기존 리스트 저장 흐름 유지 =====
            l = obs.ego_vehicle_state.lane_index
            s_list.append(t)  # s 값 대신 step index 사용 (기존 lane_position.s 대신)
            l_list.append(l)
            offset_list.append(abs(lateral_offset))
            v_list.append(ego_state.speed)
            steer_list.append(a[-1])

            if human.slow_down:
                time.sleep(1/40)

            if done:
                if not info[AGENT_ID]['env_obs'].events.off_road and \
                   not info[AGENT_ID]['env_obs'].events.collisions:
                    success += 1

                # ✅ 기존 출력문 그대로 유지
                print('\n|Epoc:', ep,
                      '\n|Step:', t,
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|R:', reward_total,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)

                # ✅ CSV 저장도 그대로
                df["s"] = s_list
                df["l"] = l_list
                df["v"] = v_list
                df["steer"] = steer_list
                df.to_csv(f'./store/{env_name}/data_{name}_{ep}.csv', index=0)

                break

            reward_total += r

        ep += 1
        v_list_avg.append(np.mean(v_list))
        offset_list_avg.append(np.mean(offset_list))
        dist_list_avg.append(curr_pos[0] - initial_pos[0])
        avg_reward_list.append(reward_total)

        print("\n..............................................")
        print("%i Loop, Steps: %i, Avg Reward: %f, Success No. : %i " %
              (ep, t, reward_total, success))
        print("..............................................")

    reward = statistics.mean(avg_reward_list)
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward:%f, Success No.: %i" %
          (eval_episodes, ep, reward, success))
    print("..............................................")

    return reward, v_list_avg, offset_list_avg, dist_list_avg, avg_reward_list

# observation space
def observation_adapter(env_obs):
    global states

    # 관측 상태 비워두기 또는 기본값
    states = np.zeros(shape=(screen_size, screen_size, 9), dtype=np.uint8)

    return states

# reward function
import numpy as np

# ✅ 안전하게 행동 ID 추출하는 함수
def extract_action_id(x):
    if isinstance(x, (int, np.integer, float, np.floating)):
        return int(x)
    elif isinstance(x, np.ndarray):
        if x.size == 1:
            return int(x.item())
        elif x.ndim == 1:
            return int(np.argmax(x))  # 확률 벡터로 간주
        else:
            raise ValueError(f"Unsupported array shape: {x.shape}")
    else:
        raise TypeError(f"Unsupported type for action: {type(x)}")

# ✅ 보상 함수
def reward_adapter(env_obs, pos_list, action, engage=False, done=False, llm_action=None):
    ego_obs = env_obs.ego_vehicle_state
    ego_pos = ego_obs.position[:2]
    lane_name = ego_obs.lane_id
    lane_id = ego_obs.lane_index
    ref = env_obs.waypoint_paths

    ##### For Scratch ######
    heuristic = ego_obs.speed * 0.01

    ###### Terminal Reward #######
    if done and not env_obs.events.reached_max_episode_steps and \
       not env_obs.events.off_road and not bool(len(env_obs.events.collisions)):
        print('Good Job!')
        goal = 3.0
    else:
        goal = 0.0

    if env_obs.events.off_road:
        print('\n Off Road!')
        off_road = -7.0
    else:
        off_road = 0.0

    if env_obs.events.collisions:
        print('\n crashed')
        crash = -7.0
    else:
        crash = 0.0

    if engage and PENALTY_GUIDANCE:
        guidance = 0.0
    else:
        guidance = 0.0

    ###### Performance Penalty ######
    if len(ref[lane_id]) > 1:
        ref_pos_1st = ref[lane_id][0].pos
        ref_pos_2nd = ref[lane_id][1].pos
        ref_dir_vec = ref_pos_2nd - ref_pos_1st
        lat_error = signedDistToLine(ego_pos, ref_pos_1st, ref_dir_vec)

        ref_heading = ref[lane_id][0].heading
        ego_heading = ego_obs.heading
        heading_error = ref_heading - ego_heading
    else:
        lat_error = 0.0
        heading_error = 0.0

    performance = -0.01 * lat_error**2 - 0.1 * heading_error**2

    ##### Shoulder Penalty #####
    if env_obs.events.on_shoulder:
        print('\n on_shoulder')
        performance -= 0.1

    reward = heuristic + off_road + crash + performance + guidance + goal

    ##### ✅ LLM과 DRL 행동 일치 시 보상 추가 #####
    LLM_MATCH_REWARD = 1.0
    if llm_action is not None:
        try:
            if extract_action_id(llm_action) == extract_action_id(action):
                reward += LLM_MATCH_REWARD
        except Exception as e:
            print("[WARN] LLM action 비교 중 오류:", e)

    return reward


# action space
def action_adapter(action): 

    long_action = action[0]
    if long_action < 0:
        throttle = 0.0
        braking = abs(long_action)
    else:
        throttle = abs(long_action)
        braking = 0.0
 
    steering = action[1]

    return (throttle, braking, steering)

# information

def info_adapter(observation, reward, info):
    return info

def construct_sce(obs):
    ego = obs.ego_vehicle_state

    lane_offset = ego.position[0]

    return {
        "ego_speed": ego.speed,
        "lane_offset": lane_offset,
        "num_nearby_vehicles": len(obs.neighborhood_vehicle_states),
        "acceleration": 0.0  # 또는 나중에 계산할 수 있게 패딩
    }


def interaction(COUNTER, start_epoc=1):
    save_threshold = 3.0
    trigger_reward = 3.0
    trigger_epoc = 1
    saved_epoc = 1
    epoc = start_epoc
    pbar = tqdm(total=MAX_NUM_EPOC, initial=start_epoc)
    
    while epoc <= MAX_NUM_EPOC:
        reward_total = 0.0 
        error = 0.0 
        action_list = deque(maxlen=1)
        action_list.append(np.array([0.0, 0.0]))
        guidance_count = int(0)
        guidance_rate = 0.0
        frame_skip = 5
        
        continuous_threshold = 100
        intermittent_threshold = 300
        
        pos_list = deque(maxlen=5)
        obs = env.reset()  # SMARTS 0.6.1: reset returns obs only
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            
            info = {}

# 🔥 여기 추가: obs가 dict면 AGENT_ID로 꺼내기
        if isinstance(obs, dict):
    # 보통 AGENT_ID는 main.py 위쪽이나 상단에 정의돼 있을 거예요.
    # 예: AGENT_ID = "Agent-007"
            obs = obs.get(AGENT_ID)
        initial_pos = obs.ego_vehicle_state.position[:2]
        pos_list.append(initial_pos)
        s = observation_adapter(obs)
        
        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break

            ##### Select and perform an action ######
            rl_a, llm_action = agent.choose_action(
                np.array(s),
                action_list[-1],
                obs=obs,
                arbitrator=arbitrator,
                current_scenario="highway/overtake",
                sce=construct_sce(obs),
                human_action=human.act() if human.intervention and epoc <= INTERMITTENT_THRESHOLD else None,
                memory_module=memory_module  # ← 반드시 추가!
            )
            guidance = False
            
            ##### Human-in-the-loop #######
            if model != 'SAC' and human.intervention and epoc <= INTERMITTENT_THRESHOLD:
                human_action = human.act()
                guidance = True
            else:
                human_a = np.array([0.0, 0.0])
            
            ###### Assign final action ######
            if guidance:
                if human_action[1] > human.MIN_BRAKE:
                    human_a = np.array([-human_action[1], human_action[-1]])
                else:
                    human_a = np.array([human_action[0], human_action[-1]])
                
                if arbitrator.shared_control and epoc > CONTINUAL_THRESHOLD:
                    rl_authority, human_authority = arbitrator.authority(obs, rl_a, human_a)
                    a = rl_authority * rl_a + human_authority * human_a
                else:
                    a = human_a
                    human_authority = 1.0 #np.array([1.0, 1.0])
                engage = int(1)
                authority = human_authority
                guidance_count += int(1)
            else:
                a = rl_a
                engage = int(0)
                authority = 0.0 
            
            ##### Interaction #####
            action = action_adapter(a)
            
            ##### Safety Mask #####
            ego_state = obs.ego_vehicle_state
            lane_id = ego_state.lane_index
            if ego_state.speed >= obs.waypoint_paths[lane_id][0].speed_limit and\
               action[0] > 0.0:
                   action = list(action)
                   action[0] = 0.0
                   action = tuple(action)
                       
            action = {AGENT_ID:action}
            next_obs, reward, done, info = env.step(action)  # SMARTS 0.6.1: step returns obs, reward, done, info

            obs = next_obs[AGENT_ID]
            s_ = observation_adapter(obs)
            if isinstance(done, dict):
                done = done.get(AGENT_ID, False)
            curr_pos = next_obs[AGENT_ID].ego_vehicle_state.position[:2]
            
           
            if env_name == 'straight' and (curr_pos[0] - initial_pos[0]) > 200:
                done = True
                print('Done')
            elif env_name == 'straight_with_turn' and (curr_pos[1] - initial_pos[1]) > 98:
                done = True
                print('Done')
            
            r = reward_adapter(next_obs[AGENT_ID], pos_list, a, engage=engage, done=done, llm_action=llm_action)
            pos_list.append(curr_pos)

            ##### Store the transition in memory ######
            agent.store_transition(s, action_list[-1], a, human_a, r,
                                   s_, a, engage, authority, done)
            
            reward_total += r
            action_list.append(a)
            s = s_
                            
            if epoc >= THRESHOLD:   
                # Train the DRL model
                agent.learn_guidence(BATCH_SIZE)

            if human.slow_down:
                time.sleep(1/40)

           

            if done:
                    epoc += 1
                    interrupt_checkpoint = {
                    'epoc': epoc,
                    'policy_state': agent.policy.state_dict(),
                    'critic_state': agent.critic.state_dict(),
                    'optimizer_policy': agent.policy_optim.state_dict(),
                    'optimizer_critic': agent.critic_optim.state_dict(),
                }
                    torch.save(
                    interrupt_checkpoint,
                    os.path.join('trained_network', env_name, f'{name}_checkpoint_interrupt.pth')
                )
                    print(f"[INFO] 💾 Interrupt checkpoint 저장 완료 (epoc={epoc})")
                    if epoc > THRESHOLD:
                            reward_list.append(max(-15.0, reward_total))
                            reward_mean_list.append(np.mean(reward_list[-10:]))
                    import pandas as pd

                    df = pd.DataFrame({
                        "epoc": list(range(start_epoc, epoc + 1)),
                        "reward": reward_list,
                        "mean_reward": reward_mean_list
                    })
                    df.to_csv(f"reward_logs/reward_trace_epoc{epoc}.csv", index=False)
                    ###### Evaluating the performance of current model ######
                    # ✅ 저장 조건 2) 보상 평균 기반 저장
                    if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                        print("Evaluating the Performance.")
                        avg_reward, _, _, _, _ = evaluate(env, agent, EVALUATION_EPOC)
                        trigger_reward = avg_reward
                        if avg_reward > save_threshold:
                            print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                            saved_epoc = epoc
                            # 보상 기반 저장
                            torch.save(agent.policy.state_dict(), os.path.join('trained_network/' + env_name,
                                      f"{name}_reward_epoc{epoc}_actornet.pkl"))
                            torch.save(agent.critic.state_dict(), os.path.join('trained_network/' + env_name,
                                      f"{name}_reward_epoc{epoc}_criticnet.pkl"))
                            # === checkpoint 저장 ===
                            checkpoint = {
                                'epoc': epoc,
                                'policy_state': agent.policy.state_dict(),
                                'critic_state': agent.critic.state_dict(),
                                'optimizer_policy': agent.policy_optim.state_dict(),
                                'optimizer_critic': agent.critic_optim.state_dict(),
                            }
                            torch.save(checkpoint, os.path.join('trained_network/' + env_name, f'{name}_checkpoint_epoc{epoc}.pth'))
                            save_threshold = avg_reward

                    always_checkpoint = {
        'epoc': epoc,
        'policy_state': agent.policy.state_dict(),
        'critic_state': agent.critic.state_dict(),
        'optimizer_policy': agent.policy_optim.state_dict(),
        'optimizer_critic': agent.critic_optim.state_dict(),
    }
                    torch.save(always_checkpoint, os.path.join('trained_network', env_name, f"{name}_checkpoint_epoc{epoc}.pth"))
                    print(f"[INFO] 💾 항상 저장: {name}_checkpoint_epoc{epoc}.pth")
                    guidance_rate = 100 * guidance_count / (t + 1)

                    print('\n|Epoc:', epoc,
                      '\n|Step:', t,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      '\n|Guidance Rate:', guidance_rate, '%',
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|R:', reward_total,
                      '\n|Temperature:', agent.alpha,
                      '\n|Reward Threshold:', save_threshold,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
                    obs = env.reset()
# info가 필요하다면 아래처럼 추가
                    info = {}
                    obs = obs[AGENT_ID]  
                    s = observation_adapter(obs) 
                    reward_total = 0
                    error = 0
                    pbar.update(1)
                    break
            
        if epoc % PLOT_INTERVAL == 0:
             plot_animation_figure(saved_epoc)
    
        

        # 💡 이어서 학습(resume training)용 예시 (필요시 활성화)
        # checkpoint = {
        #     'epoc': epoc,
        #     'policy_state': agent.policy.state_dict(),
        #     'critic_state': agent.critic.state_dict(),
        #     'optimizer_policy': agent.policy_optim.state_dict(),
        #     'optimizer_critic': agent.critic_optim.state_dict(),
        # }
        # torch.save(checkpoint, os.path.join('trained_network/' + env_name, f'{name}_checkpoint_epoc{epoc}.pth'))
        #
        # 불러올 때:
        # checkpoint = torch.load(PATH)
        # agent.policy.load_state_dict(checkpoint['policy_state'])
        # agent.critic.load_state_dict(checkpoint['critic_state'])
        # agent.policy_optim.load_state_dict(checkpoint['optimizer_policy'])
        # agent.critic_optim.load_state_dict(checkpoint['optimizer_critic'])
        # epoc = checkpoint['epoc']

    pbar.close()
    print('Complete')
    return save_threshold

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    plt.ion()
    
    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    
    ##### Individual parameters for each model ######
    model = 'SAC'
    mode_param = config[model]
    name = mode_param['name']
    POLICY_GUIDANCE = mode_param['POLICY_GUIDANCE']
    VALUE_GUIDANCE = mode_param['VALUE_GUIDANCE']
    PENALTY_GUIDANCE = mode_param['PENALTY_GUIDANCE']
    ADAPTIVE_CONFIDENCE = mode_param['ADAPTIVE_CONFIDENCE']
    
    if model != 'SAC':
        SHARED_CONTROL = mode_param['SHARED_CONTROL']
        CONTINUAL_THRESHOLD = mode_param['CONTINUAL_THRESHOLD']
        INTERMITTENT_THRESHOLD = mode_param['INTERMITTENT_THRESHOLD']
    else:
        SHARED_CONTROL = False
        
    ###### Default parameters for DRL ######
    mode = config['mode']
    ACTOR_TYPE = config['ACTOR_TYPE']
    CRITIC_TYPE = config['CRITIC_TYPE']
    LR_ACTOR = config['LR_ACTOR']
    LR_CRITIC = config['LR_CRITIC']
    TAU = config['TAU']
    THRESHOLD = config['THRESHOLD']
    TARGET_UPDATE = config['TARGET_UPDATE']
    BATCH_SIZE = config['BATCH_SIZE']
    GAMMA = config['GAMMA']
    MEMORY_CAPACITY = config['MEMORY_CAPACITY']
    MAX_NUM_EPOC = config['MAX_NUM_EPOC']
    MAX_NUM_STEPS = config['MAX_NUM_STEPS']
    PLOT_INTERVAL = config['PLOT_INTERVAL']
    SAVE_INTERVAL = config['SAVE_INTERVAL']
    EVALUATION_EPOC = config['EVALUATION_EPOC']

    ###### Entropy ######
    ENTROPY = config['ENTROPY']
    LR_ALPHA = config['LR_ALPHA']
    ALPHA = config['ALPHA']
    
    ###### Env Settings #######
    env_name = config['env_name']
    scenario = config['scenario_path']
    screen_size = config['screen_size']
    view = config['view']
    condition_state_dim = config['condition_state_dim']
    AGENT_ID = config['AGENT_ID']
    
    # Create the network storage folders
    if not os.path.exists("./store/" + env_name):
        os.makedirs("./store/" + env_name)
        
    if not os.path.exists("./trained_network/" + env_name):
        os.makedirs("./trained_network/" + env_name)

    ##### Train #####
    for i in range(1, 2):
        seed = i

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #### Environment specs ####
        ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(screen_size, screen_size, 9))
        states = np.zeros(shape=(screen_size, screen_size, 9))
    
        ##### Define agent interface #######
        agent_interface = AgentInterface(
            
            max_episode_steps=MAX_NUM_STEPS,
            waypoints=Waypoints(50),
            neighborhood_vehicles=NeighborhoodVehicles(radius=100),
           
           
            action=ActionSpaceType.Continuous,
        )
        
        ###### Define agent specs ######
        agent_spec = AgentSpec(
            interface=agent_interface,
            # observation_adapter=observation_adapter,
            # reward_adapter=reward_adapter,
            # action_adapter=action_adapter,
            # info_adapter=info_adapter,
        )
        
        ######## Human Intervention through g29 or keyboard ########
        human = HumanKeyboardAgent()
        
        ##### Create Env ######
        if model == 'SAC':
            envisionless = True
        else:
            envisionless = False
        
        scenario_path = [scenario]
        
        
        env = HiWayEnv(
    scenarios=scenario_path,
    agent_specs={AGENT_ID: agent_spec},
    headless=False,
    visdom=False,
    sumo_headless=True,
    seed=seed, 
    
)


        env.observation_space = OBSERVATION_SPACE
        env.action_space = ACTION_SPACE
        env.agent_id = AGENT_ID
        env.seed = seed

        obs = env.reset()  # SMARTS 0.6.1: reset returns obs only
        img_h, img_w, channel = screen_size, screen_size, 9
        physical_state_dim = 2
        n_obs = img_h * img_w * channel
        n_actions = env.action_space.high.size
        
        legend_bar = []
        
        # Initialize the agent
        agent = DRL(seed, n_actions, channel, condition_state_dim, ACTOR_TYPE, CRITIC_TYPE,
                    LR_ACTOR, LR_CRITIC, LR_ALPHA, MEMORY_CAPACITY, TAU,
                    GAMMA, ALPHA, POLICY_GUIDANCE, VALUE_GUIDANCE,
                    ADAPTIVE_CONFIDENCE, ENTROPY)
        
        arbitrator = Arbitrator()
        arbitrator.shared_control = SHARED_CONTROL
        
        memory_module = MemoryModule()
        
        legend_bar.append('seed'+str(seed))
        
        train_durations = []
        train_durations_mean_list = []
        reward_list = []
        reward_mean_list = []
        guidance_list = []

        
        print('\nThe object is:', model, '\n|Seed:', agent.seed, 
             '\n|VALUE_GUIDANCE:', VALUE_GUIDANCE, '\n|PENALTY_GUIDANCE:', PENALTY_GUIDANCE,'\n')
        
        success_count = 0

        # === 이어서 학습(Resume) 관련 설정 ===
        RESUME = True  # 이어서 학습하지 않고 처음부터 시작
        CHECKPOINT_PATH = "trained_network/straight/sac_speed_checkpoint_epoc821.pth"  
        start_epoc = 1

        
        #### pkl 파일 평가 코드 ####
        #### pth 파일 평가 코드 ####
        #### pkl 파일 평가 코드 ####
        if mode == 'evaluation':
                    name = 'sac'
                    environment_name = 'straight'
                    max_epoc = 820
                    max_steps = 300
                    seed = 4
                    directory = 'trained_network/' + environment_name
                    # directory =  'best_candidate'
                    filename = 'sac_speed_reward_epoc532'
                    agent.policy.load_state_dict(torch.load(f'{directory}/{filename}_actornet.pkl'))
                    agent.policy.eval()
                    reward, v_list_avg, offset_list_avg, dist_list_avg, avg_reward_list = evaluate(env, agent, eval_episodes=10)
                    
                    print('\n|Avg Speed:', np.mean(v_list_avg),
                        '\n|Std Speed:', np.std(v_list_avg),
                        '\n|Avg Dist:', np.mean(dist_list_avg),
                        '\n|Std Dist:', np.std(dist_list_avg),
                        '\n|Avg Offset:', np.mean(offset_list_avg),
                        '\n|Std Offset:', np.std(offset_list_avg))


        else:
           
            try:
                # === 이어서 학습(Resume) ===
                if RESUME and os.path.exists(CHECKPOINT_PATH):
                    checkpoint = torch.load(CHECKPOINT_PATH)
                    agent.policy.load_state_dict(checkpoint['policy_state'])
                    agent.critic.load_state_dict(checkpoint['critic_state'])
                    agent.policy_optim.load_state_dict(checkpoint['optimizer_policy'])
                    agent.critic_optim.load_state_dict(checkpoint['optimizer_critic'])
                    start_epoc = checkpoint['epoc']
                    print(f"[Resume] Checkpoint에서 epoc={start_epoc}부터 이어서 학습합니다.")

                # 🚀 그냥 interaction 호출
                save_threshold = interaction(success_count, start_epoc=start_epoc)

            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt 발생! 현재 모델을 안전 저장합니다...")
                checkpoint = {
                    'epoc': start_epoc,  # 현재 진행된 epoc을 넣어도 됩니다. interaction 내에서 최신 값 넘겨도 좋습니다.
                    'policy_state': agent.policy.state_dict(),
                    'critic_state': agent.critic.state_dict(),
                    'optimizer_policy': agent.policy_optim.state_dict(),
                    'optimizer_critic': agent.critic_optim.state_dict(),
                }
                torch.save(checkpoint, os.path.join('trained_network', env_name, f"{name}_checkpoint_interrupt.pth"))
                print("[INFO] 안전 저장 완료! 프로그램을 종료합니다.")
                torch.save(agent.policy.state_dict(),
                    os.path.join('trained_network', env_name,
                                    f"{name}_reward_epoc{start_epoc}_actornet.pkl"))
                torch.save(agent.critic.state_dict(),
                    os.path.join('trained_network', env_name,
                                    f"{name}_reward_epoc{start_epoc}_criticnet.pkl"))
                print("[INFO] ✅ Interrupt 시점 actornet/criticnet 저장 완료")

                print("[INFO] 프로그램을 종료합니다.")
