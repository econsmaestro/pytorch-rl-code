import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import gradio as gr
from PIL import Image


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = QNetwork(state_size=4, action_size=2)
model.load_state_dict(torch.load("cartpole_dqn.pth", map_location="cpu"))
model.eval()


def play_cartpole(max_steps):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
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

    # Save as GIF
    gif_path = "cartpole_play.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    return gif_path, f"Total reward: {int(total_reward)} | Steps survived: {len(frames)}"


demo = gr.Interface(
    fn=play_cartpole,
    inputs=gr.Slider(50, 500, value=200, step=50, label="Max Steps"),
    outputs=[
        gr.Image(label="Agent Playing CartPole", type="filepath"),
        gr.Textbox(label="Result"),
    ],
    title="CartPole DQN Agent",
    description="Watch a trained DQN agent balance a pole on a cart. Adjust max steps and hit Run.",
)

if __name__ == "__main__":
    demo.launch()
