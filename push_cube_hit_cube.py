import numpy as np, sapien.core as sapien, torch
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv


@register_env("PushCubeHitCube-v1", max_episode_steps=100)
class PushCubeHitCubeEnv(PushCubeEnv):

    # -----------------------------------------------------------
    # 1)  Insert cube B exactly once, right after the scene exists
    # -----------------------------------------------------------
    def _load_scene(self, options):
        super()._load_scene(options)                 # panda, table, cube A, goal
        half = 0.02
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[half]*3, density=400)
        builder.add_box_visual(half_size=[half]*3,
                               material=sapien.render.RenderMaterial(base_color=[0.2, 0.3, 0.9, 1]))
        builder.initial_pose = sapien.Pose([0, 0, half])
        self.cube_B = builder.build(name="cube_B")

    # -----------------------------------------------------------
    # 2)  Randomise cube B pose each episode
    # -----------------------------------------------------------
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
    
        # sample XY on opposite halves (torch → numpy at the end)
        xy = torch.empty((len(env_idx), 2), device=self.device).uniform_(-0.07, 0.07)
        xy[:, 1] = torch.where(torch.rand(len(env_idx), device=self.device) > 0.5,
                               0.08, -0.08)
    
        # build a NumPy (N,3) array for positions
        pos_np = torch.cat([xy, torch.full((len(env_idx), 1),
                                           0.02, device=self.device)], dim=-1).cpu().numpy()
    
        for i, idx in enumerate(env_idx.cpu().numpy()):
            self.cube_B.set_pose(sapien.Pose(p=pos_np[i]))

    # -----------------------------------------------------------
    # 3)  Reward, success flag, and episode termination
    # -----------------------------------------------------------
    def compute_reward_and_done(self):
        """
        Called every step by ManiSkill.  MUST return (reward, done, info).
        `done` should become True either when the task is solved or when a
        failure condition occurs.  ManiSkill will take care of the time-limit
        truncation separately.
        """
        # 1. distance between the two cubes in the X-Y plane
        dist = torch.norm(self.obj.pose.p[:, :2] -
                          self.cube_B.pose.p[:, :2], dim=-1)      # (N,)
            
        # 2. dense shaping reward: near-zero → 1.0
        reward = 1.0 - torch.tanh(4.0 * dist_xy)
    
        # 3. success flag
        success = dist_xy < 0.015                                    # Bool tensor
    
        # 4. optional +1 bonus when success is first reached
        reward = reward + success.float()
    
        # 5. terminate episode immediately upon success
        done = success
    
        # 6. diagnostics that land in info["final_info"]["episode"]
        info = {"is_success": success}
    
        return reward, done, info
    
    # -----------------------------------------------------------
    # 4)  Observation
    # -----------------------------------------------------------
    def get_obs(self, info=None, *args, **kwargs):
        parent_obs = super().get_obs(info, *args, **kwargs)
        def _find_tensor(d):
            """Depth-first search: return first torch.Tensor in a nested dict."""
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, dict):
                    t = _find_tensor(v)
                    if t is not None:
                        return t
            return None
        # first reset: cube_B not ready yet
        if not hasattr(self, "cube_B"):
            return parent_obs
    
        # ---------- rollout tensors ----------
        if isinstance(parent_obs, torch.Tensor):
            pose_p = torch.as_tensor(self.cube_B.pose.p, device=parent_obs.device)
            pose_q = torch.as_tensor(self.cube_B.pose.q, device=parent_obs.device)
            return torch.cat([parent_obs, pose_p, pose_q], dim=-1)
    
        # ---------- dict (unflattened=True) ----------
        if isinstance(parent_obs, dict):
            obs = dict(parent_obs)                         # shallow copy
            sample = _find_tensor(obs)
            dev = sample.device if sample is not None else torch.device("cpu")
            obs["cube_B_p"] = torch.tensor(self.cube_B.pose.p, device=dev)
            obs["cube_B_q"] = torch.tensor(self.cube_B.pose.q, device=dev)
            return obs
        # ---------- numpy (single-env CPU) ----------
        return np.concatenate([parent_obs.ravel(),
                               self.cube_B.pose.p, self.cube_B.pose.q])

    # Success predicate
    def _check_success(self):
        dist = torch.norm(self.obj.pose.p[:, :2] -          # cube A from PushCubeEnv
                          self.cube_B.pose.p[:, :2], dim=-1)
        return dist < 0.015                                 # Bool tensor (num_envs,)
    
    # Dense reward every step
    def _get_dense_reward(self):
        dist = torch.norm(self.obj.pose.p[:, :2] -
                          self.cube_B.pose.p[:, :2], dim=-1)
        return 1.0 - torch.tanh(4.0 * dist)
