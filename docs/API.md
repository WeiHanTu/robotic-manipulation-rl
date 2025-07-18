# API Documentation

## Environment

### PushCubeHitCube-v1

A custom environment that extends the base `PushCube` environment with additional complexity.

#### Environment ID
```
PushCubeHitCube-v1
```

#### Observation Space
- **Type**: `gym.spaces.Box`
- **Shape**: Varies based on observation mode
- **Description**: State information including both cubes' positions and orientations

#### Action Space
- **Type**: `gym.spaces.Box`
- **Shape**: `(7,)`
- **Description**: 7-dimensional joint delta position control

#### Reward Function
The environment provides a dense reward based on the distance between the two cubes:

```python
def compute_reward_and_done(self):
    # Distance between cubes in X-Y plane
    dist = torch.norm(self.obj.pose.p[:, :2] - self.cube_B.pose.p[:, :2], dim=-1)
    
    # Dense shaping reward: near-zero → 1.0
    reward = 1.0 - torch.tanh(4.0 * dist)
    
    # Success flag
    success = dist < 0.015
    
    # Optional +1 bonus when success is first reached
    reward = reward + success.float()
    
    return reward, success, {"is_success": success}
```

#### Success Condition
- **Criterion**: Distance between cubes < 0.015 units
- **Termination**: Episode ends immediately upon success

#### Episode Length
- **Maximum Steps**: 100
- **Early Termination**: Yes (upon success)

## Training Scripts

### train.py

Main training script implementing PPO algorithm.

#### Command Line Arguments

##### General Arguments
- `--exp_name`: Experiment name (default: auto-generated)
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA (default: True)
- `--track`: Enable Weights & Biases tracking (default: False)
- `--capture_video`: Capture training videos (default: True)
- `--save_model`: Save model checkpoints (default: True)

##### Environment Arguments
- `--env_id`: Environment ID (default: "PushCubeHitCube-v1")
- `--total_timesteps`: Total training timesteps (default: 1,000,000)
- `--num_envs`: Number of parallel environments (default: 512)
- `--num_eval_envs`: Number of evaluation environments (default: 8)
- `--control_mode`: Control mode (default: "pd_joint_delta_pos")

##### Algorithm Arguments
- `--learning_rate`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.8)
- `--gae_lambda`: GAE lambda parameter (default: 0.9)
- `--clip_coef`: PPO clipping coefficient (default: 0.2)
- `--update_epochs`: Policy update epochs (default: 4)
- `--num_minibatches`: Number of minibatches (default: 32)
- `--target_kl`: Target KL divergence (default: 0.1)
- `--ent_coef`: Entropy coefficient (default: 0.0)
- `--vf_coef`: Value function coefficient (default: 0.5)
- `--max_grad_norm`: Maximum gradient norm (default: 0.5)

#### Example Usage
```bash
python train.py \
    --env_id PushCubeHitCube-v1 \
    --total_timesteps 3000000 \
    --learning_rate 3e-4 \
    --num_envs 512 \
    --gamma 0.8 \
    --gae_lambda 0.9 \
    --clip_coef 0.2 \
    --update_epochs 4 \
    --num_minibatches 32 \
    --target_kl 0.1 \
    --save_model \
    --capture_video
```

### baseline.py

Baseline implementation for comparison with different environments.

#### Default Environment
- **Environment ID**: "PickCube-v1"
- **Other Parameters**: Same as train.py

## Agent Architecture

### Agent Class

The agent implements an actor-critic architecture with separate networks for policy and value function.

#### Methods

##### `get_value(x)`
Returns the value function estimate for the given observation.

**Parameters:**
- `x`: Observation tensor

**Returns:**
- Value function estimate

##### `get_action(x, deterministic=False)`
Returns an action for the given observation.

**Parameters:**
- `x`: Observation tensor
- `deterministic`: Whether to use deterministic action selection

**Returns:**
- Action tensor

##### `get_action_and_value(x, action=None)`
Returns action, log probability, entropy, and value for the given observation.

**Parameters:**
- `x`: Observation tensor
- `action`: Optional action tensor (if None, samples from policy)

**Returns:**
- Tuple of (action, log_prob, entropy, value)

#### Network Architecture

##### Actor Network
```
Linear(obs_dim, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, action_dim)
```

##### Critic Network
```
Linear(obs_dim, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)
```

##### Initialization
- **Weights**: Orthogonal initialization with std=√2
- **Biases**: Zero initialization
- **Action std**: Initialized to -0.5

## Environment Implementation

### PushCubeHitCubeEnv

Custom environment class extending `PushCubeEnv`.

#### Key Methods

##### `_load_scene(options)`
Loads the scene with additional cube B.

##### `_initialize_episode(env_idx, options)`
Randomizes cube B position each episode.

##### `compute_reward_and_done()`
Computes reward and termination condition.

##### `get_obs(info=None, *args, **kwargs)`
Returns observation including cube B information.

##### `_check_success()`
Checks if the task is successful.

##### `_get_dense_reward()`
Returns dense reward based on cube proximity.

## Logging and Monitoring

### Logger Class

Handles logging to TensorBoard and Weights & Biases.

#### Methods

##### `add_scalar(tag, scalar_value, step)`
Logs a scalar value.

**Parameters:**
- `tag`: Metric name
- `scalar_value`: Value to log
- `step`: Training step

##### `close()`
Closes the logger.

### Logged Metrics

- **losses/explained_variance**: Value function prediction quality
- **losses/policy_loss**: Actor network loss
- **losses/value_loss**: Critic network loss
- **train/episode_len**: Average episode length
- **train/return**: Average episode return
- **train/success_once**: Success rate

## Evaluation

### Evaluation Mode

Run evaluation with a trained model:

```bash
python train.py --evaluate --checkpoint path/to/model.pt
```

### Evaluation Metrics

- **Success Rate**: Percentage of successful episodes
- **Average Return**: Mean episode return
- **Average Episode Length**: Mean episode length
- **Video Recording**: Optional video capture of evaluation episodes

## Demo Script

### demo_random_actions.py

Simple demonstration script that runs random actions in the environment.

#### Usage
```bash
python demo_random_actions.py
```

#### Features
- Records episode video
- Saves to `videos/random_actions/`
- Demonstrates environment interaction

## File Structure

```
robotic-manipulation-rl/
├── train.py                 # Main training script
├── baseline.py              # Baseline implementation
├── push_cube_hit_cube.py   # Custom environment
├── demo_random_actions.py  # Demo script
├── requirements.txt         # Dependencies
├── setup.py                # Installation script
├── LICENSE                 # MIT License
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── docs/                  # Documentation
    └── API.md            # This file
```

## Dependencies

### Core Dependencies
- **torch**: PyTorch deep learning framework
- **gymnasium**: Reinforcement learning environment interface
- **mani-skill**: Robotic manipulation environment
- **numpy**: Numerical computing
- **tensorboard**: Training visualization
- **tyro**: Command line argument parsing

### Optional Dependencies
- **wandb**: Experiment tracking
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## Performance Benchmarks

### Training Performance
- **Convergence**: ~200k steps for initial learning
- **Final Success Rate**: >95%
- **Explained Variance**: >90%
- **Average Episode Length**: ~15-20 steps

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ for training logs and videos 