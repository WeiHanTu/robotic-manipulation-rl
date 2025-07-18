#!/usr/bin/env python3
"""
Example script demonstrating how to use the training functionality.
"""

import os
import sys
import torch
import gymnasium as gym

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import push_cube_hit_cube
from train import Agent, Args


def main():
    """
    Example of how to train a model using the custom environment.
    """
    print("🚀 Starting Robotic Manipulation RL Training Example")
    
    # Create the environment
    print("📦 Creating environment...")
    env = gym.make(
        "PushCubeHitCube-v1",
        obs_mode="state",
        render_mode="rgb_array",
        control_mode="pd_joint_delta_pos"
    )
    
    # Create the agent
    print("🧠 Creating agent...")
    agent = Agent(env)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    print(f"💻 Using device: {device}")
    
    # Example of getting an action
    print("🎯 Testing agent...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new gymnasium API
    
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    # Get action from agent
    with torch.no_grad():
        action = agent.get_action(obs_tensor, deterministic=True)
        value = agent.get_value(obs_tensor)
    
    print(f"📊 Observation shape: {obs_tensor.shape}")
    print(f"🎮 Action shape: {action.shape}")
    print(f"💰 Value estimate: {value.item():.4f}")
    
    # Example of training configuration
    print("\n⚙️  Training Configuration Example:")
    args = Args()
    print(f"   Environment: {args.env_id}")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Number of environments: {args.num_envs}")
    print(f"   Gamma: {args.gamma}")
    print(f"   GAE lambda: {args.gae_lambda}")
    print(f"   PPO clip coefficient: {args.clip_coef}")
    
    # Example command to run training
    print("\n🔧 To run training, use:")
    print("python train.py --env_id PushCubeHitCube-v1 --total_timesteps 1000000")
    
    # Example of evaluation
    print("\n📈 To evaluate a trained model:")
    print("python train.py --evaluate --checkpoint runs/model.pt")
    
    print("\n✅ Example completed successfully!")
    
    # Clean up
    env.close()


def demonstrate_environment():
    """
    Demonstrate the environment interaction.
    """
    print("\n🎮 Environment Demonstration")
    print("=" * 40)
    
    env = gym.make(
        "PushCubeHitCube-v1",
        obs_mode="state",
        render_mode="rgb_array",
        control_mode="pd_joint_delta_pos"
    )
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print(f"📊 Initial observation shape: {obs.shape}")
    print(f"🎮 Action space: {env.action_space}")
    print(f"📏 Observation space: {env.observation_space}")
    
    # Run a few random steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward = {reward:.4f}, Done = {done}")
        
        if done or truncated:
            break
    
    print(f"🏁 Episode finished. Total reward: {total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    try:
        main()
        demonstrate_environment()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1) 