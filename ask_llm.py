import numpy as np
import json
import openai

import pre_prompt

# API 키 불러오기
with open('MY_KEY.txt', 'r', encoding='utf-8-sig') as f:
    api_key = f.read().strip()
openai.api_key = api_key

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

def send_to_chatgpt(last_action, current_scenario, sce):
    print("=========================", type(last_action), "=========================")
    
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
    system_prompt = (f"{message_prefix}"
                     f"You, the 'ego' car, are now driving on a highway. You have already driven for {frame} seconds.\n"
                     "There are several rules you need to follow when you drive on a highway:\n"
                     f"{traffic_rules}\n\n"
                     "Here are your attention points:\n"
                     f"{decision_cautions}\n\n"
                     "Once you make a final decision, output it in the following format:\n"
                     "```\n"
                     "Final Answer: \n"
                     "    \"decision\": {\"<ego car's decision, ONE of the available actions>\"},\n"
                     "```\n")

    user_prompt = (f"The decision made by the agent LAST time step was `{action_name}` ({action_description}).\n\n"
                   "Here is the current scenario:\n"
                   f"{current_scenario}\n\n")

    # ChatCompletion 생성 (openai 방식)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    llm_result = completion.choices[0].message
    llm_text = llm_result.content.strip()

# ✅ 응답 결과만 출력
    print("\n🧠 [LLM 응답]")
    print(json.dumps({"role": llm_result.role, "content": llm_text}, indent=2))

# (선택) 행동 이름 추출해서 추가 출력
    import re
    match = re.search(r'"decision":\s*{["\']?(\w+)["\']?}', llm_text)
    llm_action_str = match.group(1) if match else "IDLE"
    print("llm action:", llm_action_str)
    return llm_result  

