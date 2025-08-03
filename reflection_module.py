import os
import textwrap
import time
import requests
from rich import print

class ReflectionModule:
    def __init__(self, temperature: float = 0.0, verbose: bool = False, memory_module=None):
        self.temperature = temperature
        self.verbose = verbose
        self.memory_module = memory_module

    def reflect(self, episode_log, llm_response=None):
        """
        episode_log: [(obs, action, result), ...] 리스트
        llm_response: (optional) LLM의 reasoning 결과(문자열)
        """
        print(f"[DEBUG][Reflection] reflect called with episode_log length={len(episode_log)}")
        delimiter = "####"
        # 에피소드 요약 생성
        summary = "\n".join([
            f"Obs: {str(obs)[:40]}... Action: {action}, Result: {result}" for obs, action, result in episode_log[:2]
        ] + (["..."] if len(episode_log) > 4 else []) + [
            f"Obs: {str(obs)[:40]}... Action: {action}, Result: {result}" for obs, action, result in episode_log[-2:]
        ])
        # 프롬프트 구성
        system_message = textwrap.dedent(f"""\
        You are a large language model driving assistant. You will be given a detailed description of the driving scenario of current frame along with the available actions allowed to take. 
        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 
        Make sure to include {delimiter} to separate every step.
        """)
        if llm_response is None:
            llm_response = "(예시) LLM의 reasoning 결과가 여기에 들어갑니다."
        human_message = textwrap.dedent(f"""\
            ``` Episode Log ```
            {summary}
            ``` ChatGPT Response ```
            {llm_response}

            Now, you know this action ChatGPT output cause a collison after taking this action, which means there are some mistake in ChatGPT resoning and cause the wrong action.    
            Please carefully check every reasoning in ChatGPT response and find out the mistake in the reasoning process of ChatGPT, and also output your corrected version of ChatGPT response.
            Your answer should use the following format:
            {delimiter} Analysis of the mistake:
            <Your analysis of the mistake in ChatGPT reasoning process>
            {delimiter} What should ChatGPT do to avoid such errors in the future:
            <Your answer>
            {delimiter} Corrected version of ChatGPT response:
            <Your corrected version of ChatGPT response>
        """)

        print("[ReflectionModule] Self-reflection is running, may take time...")
        start_time = time.time()
        # Ollama API 호출
        prompt = system_message + "\n" + human_message
        url = "http://localhost:11434/api/generate"
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

        target_phrase = f"{delimiter} What should ChatGPT do to avoid such errors in the future:"
        substring = ""
        if target_phrase in llm_text:
            substring = llm_text[llm_text.find(target_phrase)+len(target_phrase):].strip()
        corrected_memory = f"{delimiter} I have made a mistake before and below is my self-reflection:\n{substring}"
        print("[ReflectionModule] Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))
        print("[ReflectionModule] corrected_memory:", corrected_memory)
        print(f"[DEBUG][Reflection] reflect result: {corrected_memory}")

        # MemoryModule에 피드백 저장 (옵션)
        if self.memory_module is not None:
            self.memory_module.save("reflection_feedback", "-", corrected_memory)

        return corrected_memory 