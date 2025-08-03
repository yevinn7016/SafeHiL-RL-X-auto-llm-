import torch

ckpt_path = r"C:\Users\yebin\PycharmProjects\Project\safehil-rl X autollm\SafeHiL-RL-X-auto-llm-\trained_network\straight\sac_speed_checkpoint_interrupt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("이 checkpoint에 저장된 epoc 값:", ckpt.get('epoc', '❌ epoc 키 없음'))
