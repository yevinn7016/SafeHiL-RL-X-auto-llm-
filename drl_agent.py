#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:12:28 2023

@author: oscar
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import re
import time


from utils_ import soft_update, hard_update
from Network import GaussianPolicy, QNetwork, DeterministicPolicy
from cpprb import PrioritizedReplayBuffer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../In-context_Learning_for_Automated_Driving/Auto_Driving_Highway'))
from ask_llm import send_to_gemini

# --- [추가] LLM 이산 action ID를 연속 action 벡터로 매핑하는 테이블 ---
ACTIONS_ID_TO_VEC = {
    0: np.array([-1.0, 0.0]),   # LANE_LEFT: 좌로 조향
    1: np.array([0.0, 0.0]),    # IDLE: 정지/유지
    2: np.array([1.0, 0.0]),    # LANE_RIGHT: 우로 조향
    3: np.array([0.0, 1.0]),    # FASTER: 가속
    4: np.array([0.0, -1.0]),   # SLOWER: 감속
}

class DRL(object):
    def __init__(self, seed, action_dim, state_dim, pstate_dim, policy_type, critic_type, 
                 LR_A = 1e-3, LR_C = 1e-3, LR_ALPHA=1e-4, BUFFER_SIZE=int(2e5), 
                 TAU=5e-3, GAMMA = 0.99, ALPHA=0.05, POLICY_GUIDANCE=False,
                 VALUE_GUIDANCE = False, ADAPTIVE_CONFIDENCE = True,
                 automatic_entropy_tuning=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.pstate_dim = pstate_dim
        self.policy_type = policy_type
        self.critic_type = critic_type
        self.gamma = GAMMA
        self.lr_a = LR_A
        self.lr_c = LR_C
        self.lr_alpha = LR_ALPHA
        self.tau = TAU
        self.alpha = ALPHA
        self.itera = 0
        self.demonstration_weight = 1.0
        self.engage_weight = 1.0 #IARL 2.0, others 1.0
        self.buffer_size_expert = 5e3
        self.p_guidance = POLICY_GUIDANCE
        self.v_guidance = VALUE_GUIDANCE
        self.p_loss = 0.0
        self.engage_loss = 0.0
        self.adaptive_weight = ADAPTIVE_CONFIDENCE
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.seed = int(seed)

        self.pre_buffer = False
        self.batch_expert = 0

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE,
                                          {"obs": {"shape": (100,100,9),"dtype": np.uint8},
                                           "pobs": {"shape":pstate_dim},
                                           "act": {"shape":action_dim},
                                           "act_h" : {"shape":action_dim},
                                           "rew": {},
                                           "next_obs": {"shape": (100,100,9),"dtype": np.uint8},
                                           "next_pobs": {"shape":pstate_dim},
                                           "engage": {},
                                           "authority": {},
                                           "done": {}},
                                          next_of=("obs"))

        ######## Initialize Critic Network #########
        self.critic = QNetwork(self.action_dim, self.state_dim, self.pstate_dim).to(device=self.device)
        self.critic_target = QNetwork(self.action_dim, self.state_dim, self.pstate_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_c)
        
        ######### target network update ########
        hard_update(self.critic_target, self.critic)

        ######### Initialize Actor Network #######
        self.policy = GaussianPolicy(self.action_dim, self.state_dim, self.pstate_dim).to(self.device)
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = - self.action_dim
            # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.log_alpha = torch.tensor(np.array(np.log(self.alpha )), requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr_alpha)
        
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr_a)

    def choose_action(self, istate, pstate, obs=None, arbitrator=None, current_scenario=None, sce=None, human_action=None, evaluate=False, llm_conf_default=0.3, memory_module=None):
        """
        DRL, LLM, Human의 행동을 받아 동적 가중치(권한)로 최종 행동을 결정
        - arbitrator.authority()로 DRL/Human 가중치 계산
        - LLM confidence 기반으로 LLM 가중치 계산
        - 세 weight는 합이 1이 되도록 정규화
        - current_scenario, sce가 없으면 LLM 미호출
        - human_action 없으면 DRL action 사용
        - evaluate 모드 유지
        """
        # 1. DRL action
        if istate.ndim < 4:
            istate_t = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
            pstate_t = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate_t = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate_t = torch.FloatTensor(pstate).float().to(self.device)
        if evaluate is False:
            drl_action, _, _ = self.policy.sample([istate_t, pstate_t])
        else:
            _, _, drl_action = self.policy.sample([istate_t, pstate_t])
        drl_action = drl_action.detach().squeeze(0).cpu().numpy()
# 2. Memory에서 유사 사례 검색 (LLM 프롬프트에 활용)
        similar_cases = []
        if memory_module is not None and obs is not None:
            print(f"[DEBUG][choose_action] step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane_id={obs.ego_vehicle_state.lane_id}")
            similar_cases = memory_module.retrieve_similar_cases(obs, k=3)

    # ✅ 간략 출력
            #print(f"[choose_action] LLM 프롬프트에 포함될 유사 사례 ({len(similar_cases)}건):")
            for idx, case in enumerate(similar_cases, 1):
        # 문자열로 된 obs에서 step/speed/lane 같은 값 파싱 필요시 추가
        # 여기서는 action만 간략히 출력
                action_str = case.get("action", "?")
        # step과 speed, lane은 obs 안에 포함 → 필요시 추출
                print(f"  {idx}) action={action_str}")

        # 2. Human action (입력값이 없으면 DRL+LLM 동적 가중치 결합)
        if human_action is None:
            # --- [변경] Human 개입이 없는 경우: DRL과 LLM 행동을 cosine similarity 기반으로 안전하게 결합 ---
            llm_result = None
            llm_action = drl_action
            llm_conf = llm_conf_default

            if current_scenario is not None and sce is not None:
                
                llm_result = send_to_gemini(drl_action, current_scenario, sce)
                time.sleep(0.1)  
                

            # ==== LLM 응답 간략 출력 ====
            if llm_result is not None and "content" in llm_result:
                llm_text = llm_result["content"]
    # JSON에서 decision만 추출
                match = re.search(r'"decision":\s*{["\']?([^"\'}]+)["\']?}', llm_text)
                llm_action_str = match.group(1) if match else "IDLE"
                print("🧠 [LLM 응답] (결정만 출력)")
                print(f"llm action: {llm_action_str}")
            else:
                #print("[ERROR] llm_result is None or missing 'content' key")
                llm_text = ""



            match = re.search(r'"decision":\s*{["\']?([^"\'}]+)["\']?}', llm_text)
            llm_action_str = match.group(1) if match else "IDLE"
            ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
            llm_action = [k for k, v in ACTIONS_ALL.items() if v == llm_action_str]
            llm_action = llm_action[0] if llm_action else 1  # 기본값 IDLE
            conf_match = re.search(r'"confidence":\s*([0-9.]+)', llm_text)
            if conf_match:
                llm_conf = float(conf_match.group(1))

            # --- [변경] LLM action을 DRL action space에 맞는 벡터로 변환 ---
            llm_action_vec = ACTIONS_ID_TO_VEC.get(int(llm_action), np.array([0.0, 0.0]))
            # --- [변경] DRL과 LLM 행동의 cosine similarity 계산 (동일 차원) ---
            drl_vec = np.array(drl_action).flatten()
            llm_vec = llm_action_vec.flatten()
            if drl_vec.shape != llm_vec.shape:
                raise ValueError(f"Cannot align drl_action shape {drl_vec.shape} and llm_action_vec shape {llm_vec.shape}")
            if np.linalg.norm(drl_vec) < 1e-8 or np.linalg.norm(llm_vec) < 1e-8:
                cosine_sim = 0.0
            else:
                cosine_sim = np.dot(drl_vec, llm_vec) / (np.linalg.norm(drl_vec) * np.linalg.norm(llm_vec))
            cosine_sim = max(0.0, cosine_sim)

            threshold = 0.5  # 코사인 유사도 임계값
            if llm_conf <= 0.2 or cosine_sim < threshold:
                # LLM 신뢰도가 매우 낮거나, 행동 방향이 충분히 유사하지 않으면 DRL 100% 사용
                rl_weight = 1.0
                llm_weight = 0.0
            else:
                # LLM confidence와 cosine similarity를 곱해 LLM 가중치로 사용
                llm_weight = llm_conf * cosine_sim
                rl_weight = 1 - llm_weight
            # 가중치 합이 1이 되도록 보장
            total = rl_weight + llm_weight
            rl_weight = rl_weight / total
            llm_weight = llm_weight / total
            human_weight = 0.0
            # --- [변경] DRL/LLM 가중 평균으로 최종 행동 산출 ---
            final_action = rl_weight * drl_vec + llm_weight * llm_vec
            if isinstance(final_action, float) or isinstance(final_action, np.floating):
                final_action = int(round(final_action))
            else:
                final_action = np.round(final_action).astype(float)  # 연속 action space 유지
            # --- [추가] Memory에 현재 step 저장 (Human 개입 여부와 무관하게) ---
            if memory_module is not None and obs is not None:
    # N step마다 저장
                N = 10  # 원하는 step 간격
                if obs.step_count % N == 0:
                    #print(f"[DEBUG][choose_action] calling memory_module.save | "
                        #f"step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane={obs.ego_vehicle_state.lane_id}")
                    
                    memory_module.save(obs, final_action, "-")
                    
                        
            return final_action, llm_action
        # --- [기존] Human 개입이 있는 경우: DRL/Human/LLM 가중치 결합 ---
        if arbitrator is not None and obs is not None:
            rl_weight_raw = arbitrator.authority(obs, drl_action, human_action)
            human_weight_raw = 1 - rl_weight_raw
        else:
            rl_weight_raw, human_weight_raw = 0.5, 0.5
        last_action = drl_action
      
        llm_result = send_to_gemini(last_action, current_scenario, sce)
        time.sleep(0.1)  
        # 2. Human action (입력값이 없으면 DRL+LLM 동적 가중치 결합)
        if human_action is None:
            # --- [변경] Human 개입이 없는 경우: DRL과 LLM 행동을 cosine similarity 기반으로 안전하게 결합 ---
            llm_result = None
            llm_action = drl_action
            llm_conf = llm_conf_default

            if current_scenario is not None and sce is not None:
                
                llm_result = send_to_gemini(drl_action, current_scenario, sce)
                time.sleep(0.1)  
                
        # 4. LLM action 및 confidence 계산
        llm_result = None
        llm_action = drl_action
        llm_conf = llm_conf_default

        if current_scenario is not None and sce is not None:
          
            llm_result = send_to_gemini(drl_action, current_scenario, sce)
            time.sleep(0.1)  

        if llm_result is not None and "content" in llm_result:
            llm_text = llm_result["content"]
    # 정규표현식으로 decision/LLM confidence 추출
        else:
            #print("[ERROR] llm_result is None or missing 'content' key:", llm_result)
            llm_text = ""


            match = re.search(r'"decision":\s*{["\']?([^"\'}]+)["\']?}', llm_text)
            llm_action_str = match.group(1) if match else "IDLE"
            ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
            llm_action = [k for k, v in ACTIONS_ALL.items() if v == llm_action_str]
            llm_action = llm_action[0] if llm_action else 1  # 기본값 IDLE
            # confidence 추출 (예: "confidence": 0.7)
            conf_match = re.search(r'"confidence":\s*([0-9.]+)', llm_text)

            if conf_match:
                llm_conf = float(conf_match.group(1))
        # LLM 가중치
        llm_weight = llm_conf

        # 5. 세 weight 정규화
        total = rl_weight_raw + human_weight_raw + llm_weight
        if      total > 0:
            rl_weight = rl_weight_raw / total
            human_weight = human_weight_raw / total
            llm_weight = llm_weight / total
        else:
            rl_weight, human_weight, llm_weight = 0.4, 0.4, 0.2

        # 6. 최종 행동 산출 (이산 action space 기준)
        final_action = (
            rl_weight * drl_action +
            llm_weight * llm_action +
            human_weight * human_action
        )
        if isinstance(final_action, float) or isinstance(final_action, np.floating):
            final_action = int(round(final_action))
        else:
            final_action = np.round(final_action).astype(int)

        # --- [추가] Memory에 현재 step 저장 (Human 개입 여부와 무관하게) ---
        if memory_module is not None and obs is not None:
            print(f"[DEBUG][choose_action] step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane_id={obs.ego_vehicle_state.lane_id}")
            
        return final_action, llm_action

    def learn_guidence(self, batch_size=64):
        agent_buffer_size = self.replay_buffer.get_stored_size()

        ###### For prior demonstration ######
        if self.pre_buffer:
            exp_buffer_size = self.replay_buffer_expert.get_stored_size()
            scale_factor = 1
            # total_size = agent_buffer_size + exp_buffer_size
            
            self.batch_expert = min(np.floor(exp_buffer_size/agent_buffer_size * batch_size / scale_factor), batch_size)

            batch_agent = batch_size
        
        if self.batch_expert > 0:
            demonstration_flag = True
            data_agent = self.replay_buffer.sample(batch_agent)
            data_expert = self.replay_buffer_expert.sample(self.batch_expert)

            istates_agent, pstates_agent, actions_agent, engages = \
                data_agent['obs'], data_agent['pobs'], data_agent['act'], data_agent['engage']
            rewards_agent, next_istates_agent, next_pstates_agent, dones_agent = \
                data_agent['rew'], data_agent['next_obs'], data_agent['next_pobs'], data_agent['done']

            istates_expert, pstates_expert, actions_expert = \
                data_expert['obs'], data_expert['pobs'], data_expert['act_exp']
            rewards_expert, next_istates_expert, next_pstates_expert, dones_expert = \
                data_expert['rew'], data_expert['next_obs'], data_expert['next_pobs'], data_expert['done']

            istates = np.concatenate((istates_agent, istates_expert), axis=0)
            pstates = np.concatenate([pstates_agent, pstates_expert], axis=0)
            actions = np.concatenate([actions_agent, actions_expert], axis=0)
            rewards = np.concatenate([rewards_agent, rewards_expert], axis=0)
            next_istates = np.concatenate([next_istates_agent, next_istates_expert], axis=0)
            next_pstates = np.concatenate([next_pstates_agent, next_pstates_expert], axis=0)
            dones = np.concatenate([dones_agent, dones_expert], axis=0)

        ###### For non prior demonstration #####
        else:
            demonstration_flag = False
            data = self.replay_buffer.sample(batch_size)
            istates, pstates, actions = data['obs'], data['pobs'], data['act']
            rewards, next_istates = data['rew'], data['next_obs']
            next_pstates, dones = data['next_pobs'], data['done']
            human_actions, engages, authorities = data['act_h'], data['engage'], data['authority']
        
        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        engages = torch.FloatTensor(engages).to(self.device)
        human_actions = torch.FloatTensor(human_actions).to(self.device)
        authorities =  torch.FloatTensor(authorities).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        ####### Critic Loss Function ######
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        ##### Pre buffer demonstration loss #####
        if demonstration_flag:
            istates_expert = torch.FloatTensor(istates_expert).permute(0,3,1,2).to(self.device)
            pstates_expert = torch.FloatTensor(pstates_expert).to(self.device)
            actions_expert = torch.FloatTensor(actions_expert).to(self.device)
            _, _, predicted_actions = self.policy.sample([istates_expert, pstates_expert]) 
            demonstration_loss = self.demonstration_weight * F.mse_loss(predicted_actions, actions_expert).mean()
        else:
            demonstration_loss = 0.0

        ####### Policy Loss Function #######
        pi, log_pi, _ = self.policy.sample([istates, pstates])
        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi).mean()
        self.p_loss = min_qf_pi.detach().cpu().numpy()
        
        ##### Human-in-the-loop loss ######
        if self.p_guidance:
            engage_index = (engages == 1).nonzero(as_tuple=True)[0]
            if engage_index.numel() > 0:
                istates_shared = istates[engage_index]
                pstates_shared = pstates[engage_index]
                target_actions = human_actions[engage_index]
                authority_shared = authorities[engage_index]
                _, _, predicted_actions = self.policy.sample([istates_shared, pstates_shared]) 
                engage_loss = F.mse_loss(predicted_actions, target_actions)
                
                if self.adaptive_weight:
                    engage_loss = (authority_shared * engage_loss).mean() * self.engage_weight
                else:
                    engage_loss = (engage_loss).mean() * self.engage_weight
                
                self.engage_loss = engage_loss.detach().cpu().numpy()
                
                policy_loss = ((self.alpha * log_pi).mean() - min_qf_pi) +\
                                  demonstration_loss + engage_loss
            else:
                authority_shared = 0.0
                engage_loss = 0.0
                policy_loss = (self.alpha * log_pi).mean() - min_qf_pi

        else:
            policy_loss = (self.alpha * log_pi).mean() - min_qf_pi

        print('P loss: ', -self.p_loss, 'Entropy Loss: ',
              (self.alpha * log_pi).mean().detach().cpu().numpy(),
              ', Engage loss: ', self.engage_loss)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Entropy Loss Function #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        ###### Update target network #####
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1
        return qf1_loss.item(), policy_loss.item()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates = data['rew'], data['next_obs']
        next_pstates, dones = data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
                
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        ###### Update target network ######        
        soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        return qf1_loss.item(), policy_loss.item()
    
    # Define the storing by priority experience replay
    def store_transition(self, s, ps, a, a_h, r, s_, ps_, e, at, d):
        self.replay_buffer.add(obs=s,
                               pobs=ps,
                               act=a,
                               act_h=a_h,
                               rew=r,
                               next_obs=s_,
                               next_pobs=ps_,
                               engage=e,
                               authority=at,
                               done=d)

    # Save and load model parameters
    def load_model(self, output):
        if output is None: return
        self.policy.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.policy.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save(self, filename, directory, reward, seed):
        torch.save(self.policy.state_dict(), '%s/%s_reward%s_seed%s_actor.pth' % (directory, filename, reward, seed))
        torch.save(self.critic.state_dict(), '%s/%s_reward%s_seed%s_critic.pth' % (directory, filename, reward, seed))

    def load(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))    

    def load_target(self):
        hard_update(self.critic_target, self.critic)

    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
