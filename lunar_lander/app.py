import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import gradio as gr
from PIL import Image


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = QNetwork(state_size=8, action_size=4)
model.load_state_dict(torch.load("lunarlander_dqn.pth", map_location="cpu"))
model.eval()


def play_lunarlander(max_steps):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    total_reward = 0

    for _ in range(int(max_steps)):
        frame = env.render()
        frames.append(Image.fromarray(frame))
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = int(model(state_t).argmax().item())
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()

    gif_path = "lunarlander_play.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)

    result = "Landed successfully!" if total_reward >= 200 else "Crashed or ran out of time."
    return gif_path, f"{result} | Total reward: {total_reward:.1f} | Steps: {len(frames)}"


demo = gr.Interface(
    fn=play_lunarlander,
    inputs=gr.Slider(100, 1000, value=500, step=100, label="Max Steps"),
    outputs=[
        gr.Image(label="Agent Playing LunarLander", type="filepath"),
        gr.Textbox(label="Result"),
    ],
    title="LunarLander DQN Agent",
    description="Watch a trained DQN agent attempt to land a spacecraft. A score above 200 means a successful landing.",
)

if __name__ == "__main__":
    demo.launch(server_port=7864)
