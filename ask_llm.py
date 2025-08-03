import numpy as np
import json
import requests
import os

import pre_prompt

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'change lane to the left of the current lane,',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle'
}

def send_to_llm_ollama(last_action, current_scenario, sce):
    #print("=========================", type(last_action), "=========================")
    
    # ✅ DRL 행동 (last_action) 출력
    try:
        if isinstance(last_action, np.ndarray):
            if last_action.size == 1:
                drl_action = int(last_action.item())
            elif last_action.ndim == 1:
                drl_action = int(np.argmax(last_action))
            else:
                raise ValueError(f"Unsupported shape for last_action: {last_action.shape}")
        else:
            drl_action = int(last_action)
    except Exception as e:
        print("[ERROR] Failed to interpret last_action:", e)
        drl_action = 1  # fallback to IDLE

    print("DQN/DRL action:", ACTIONS_ALL.get(drl_action, "Unknown"))

    message_prefix = pre_prompt.SYSTEM_MESSAGE_PREFIX
    traffic_rules = pre_prompt.get_traffic_rules()
    decision_cautions = pre_prompt.get_decision_cautions()
    action_name = ACTIONS_ALL.get(drl_action, "Unknown Action")
    action_description = ACTIONS_DESCRIPTION.get(drl_action, "No description available")

    frame = sce.get("frame", "Unknown")
    # 시스템 프롬프트 구성
    system_prompt = (
    f"{message_prefix}"
    f"You, the 'ego' car, are now driving on a highway. You have already driven for {frame} seconds.\n"
    "There are several rules you need to follow when you drive on a highway:\n"
    f"{traffic_rules}\n\n"
    "Here are your attention points:\n"
    f"{decision_cautions}\n\n"
    "Available actions you can choose from are strictly limited to this list:\n"
    "[LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]\n\n"
    "Important:\n"
    "- Do not answer in Chinese or any language other than English.\n"
    "- Only output exactly one of the available actions.\n"
    "- Do not output explanations or anything else.\n\n"
    "Once you make a final decision, output it in the following exact format:\n"
    "```\n"
    "Final Answer:\n"
    "    \"decision\": {\"<ONE of: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER>\"},\n"
    "```"
)


    user_prompt = (f"The decision made by the agent LAST time step was `{action_name}` ({action_description}).\n\n"
                   "Here is the current scenario:\n"
                   f"{current_scenario}\n\n")

    # Ollama API 호출
    prompt = system_prompt + "\n" + user_prompt
    url = "http://172.24.128.1:11434/api/generate"
    payload = {
        "model": "qwen2:7b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        llm_text = result.get("response", "").strip()
    except Exception as e:
        print("[ERROR] Ollama API 호출 실패:", e)
        llm_text = ""
        result = {"response": ""}

    print("\n🧠 [LLM 응답 - 요약]")

    import re
    match = re.search(r'"decision":\s*{["\']?(\w+)["\']?}', llm_text)
    llm_action_str = match.group(1) if match else "IDLE"
    print("llm action:", llm_action_str)
    return result

# 기존 send_to_gemini 함수와 동일한 인터페이스를 위해 alias 제공
send_to_gemini = send_to_llm_ollama 

