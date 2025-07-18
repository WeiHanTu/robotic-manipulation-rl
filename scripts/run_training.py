#!/usr/bin/env python3
"""
Convenient script to run training with different configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🚀 {description}")
    print(f"📝 Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  {description} interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run robotic manipulation RL training")
    parser.add_argument("--mode", choices=["quick", "full", "baseline", "custom"], 
                       default="quick", help="Training mode")
    parser.add_argument("--env", default="PushCubeHitCube-v1", 
                       help="Environment ID")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total timesteps")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Use GPU")
    parser.add_argument("--video", action="store_true", default=True,
                       help="Capture videos")
    parser.add_argument("--save", action="store_true", default=True,
                       help="Save model checkpoints")
    
    args = parser.parse_args()
    
    print("🤖 Robotic Manipulation RL Training Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("train.py"):
        print("❌ Error: train.py not found. Please run this script from the project root.")
        return 1
    
    # Create necessary directories
    os.makedirs("runs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    # Build command based on mode
    cmd = ["python", "train.py"]
    
    if args.mode == "quick":
        # Quick training for testing
        cmd.extend([
            "--env_id", args.env,
            "--total_timesteps", "500000",
            "--num_envs", "256",
            "--learning_rate", "3e-4",
            "--seed", str(args.seed)
        ])
        description = "Quick training (500k steps)"
        
    elif args.mode == "full":
        # Full training
        cmd.extend([
            "--env_id", args.env,
            "--total_timesteps", str(args.timesteps),
            "--num_envs", "512",
            "--learning_rate", "3e-4",
            "--gamma", "0.8",
            "--gae_lambda", "0.9",
            "--clip_coef", "0.2",
            "--update_epochs", "4",
            "--num_minibatches", "32",
            "--target_kl", "0.1",
            "--ent_coef", "0.0",
            "--vf_coef", "0.5",
            "--max_grad_norm", "0.5",
            "--seed", str(args.seed)
        ])
        description = f"Full training ({args.timesteps:,} steps)"
        
    elif args.mode == "baseline":
        # Baseline training
        cmd = ["python", "baseline.py"]
        cmd.extend([
            "--env_id", "PickCube-v1",
            "--total_timesteps", str(args.timesteps),
            "--seed", str(args.seed)
        ])
        description = "Baseline training (PickCube environment)"
        
    elif args.mode == "custom":
        # Custom training with user parameters
        cmd.extend([
            "--env_id", args.env,
            "--total_timesteps", str(args.timesteps),
            "--seed", str(args.seed)
        ])
        description = f"Custom training ({args.timesteps:,} steps)"
    
    # Add common flags
    if args.gpu:
        cmd.append("--cuda")
    else:
        cmd.append("--no-cuda")
    
    if args.video:
        cmd.append("--capture_video")
    
    if args.save:
        cmd.append("--save_model")
    
    # Run the training
    success = run_command(cmd, description)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Check the following directories for results:")
        print("   - runs/: Training logs and model checkpoints")
        print("   - videos/: Training and evaluation videos")
        print("   - TensorBoard logs: tensorboard --logdir runs/")
        
        # Show how to evaluate
        print("\n📈 To evaluate the trained model:")
        print("   python train.py --evaluate --checkpoint runs/[model_path]/model.pt")
        
        return 0
    else:
        print("\n❌ Training failed!")
        return 1

def run_multiple_seeds():
    """Run training with multiple random seeds."""
    print("🎲 Running training with multiple random seeds...")
    
    seeds = [1, 42, 123, 456, 789]
    
    for seed in seeds:
        print(f"\n🌱 Training with seed {seed}")
        cmd = [
            "python", "train.py",
            "--env_id", "PushCubeHitCube-v1",
            "--total_timesteps", "1000000",
            "--num_envs", "256",
            "--seed", str(seed),
            "--cuda",
            "--capture_video",
            "--save_model"
        ]
        
        success = run_command(cmd, f"Training with seed {seed}")
        if not success:
            print(f"❌ Failed to train with seed {seed}")
            break
    
    print("\n🎯 Multi-seed training completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        run_multiple_seeds()
    else:
        exit(main()) 