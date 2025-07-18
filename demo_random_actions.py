import gymnasium as gym
import torch
from mani_skill.utils.wrappers.record import RecordEpisode
import push_cube_hit_cube

def main():
    env = gym.make(
        "PushCubeHitCube-v1",
        render_mode="rgb_array",
        obs_mode="state",
        control_mode="pd_joint_delta_pos"
    )
    env = RecordEpisode(env, output_dir="videos/random_actions", save_video=True, video_fps=30)
    
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished. Video saved to videos/random_actions")
            break
    env.close()

if __name__ == "__main__":
    main()
