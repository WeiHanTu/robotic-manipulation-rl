# Training Figures

This directory contains training plots and visualizations from the robotic manipulation RL project.

## Expected Figures

The following figures should be generated during training and placed in the `png/` subdirectory:

### Training Performance Plots
- `losses_explained_variance_3M.png` - Explained variance progression over 3M training steps
- `losses_policy_loss_3M.png` - Policy loss over training
- `losses_value_loss_3M.png` - Value function loss over training
- `train_episode_len_3M.png` - Episode length progression
- `train_return_3M.png` - Training return over time
- `train_success_once_3M.png` - Success rate over training

### Multi-Run Comparison Plots
- `losses_explained_variance_multi.png` - Explained variance across multiple runs
- `train_return_multi.png` - Training return across multiple runs
- `train_success_once_multi.png` - Success rate across multiple runs

## Generating Figures

To generate these figures, run the training scripts:

```bash
# Generate training plots
python train.py --env_id PushCubeHitCube-v1 --total_timesteps 3000000

# Generate multi-run plots
python scripts/run_training.py multi
```

## Figure Descriptions

### Explained Variance
Shows how well the value function predicts the actual returns. Higher values indicate better value function learning.

### Policy Loss
Measures the actor network loss during PPO updates. Shows policy learning progress.

### Value Loss
Measures the critic network loss. Indicates value function learning quality.

### Episode Length
Shows how efficiently the agent completes tasks. Decreasing length indicates improved performance.

### Training Return
Average episode returns over time. Shows overall learning progress.

### Success Rate
Percentage of successful episodes. Key metric for task completion.

## Multi-Run Analysis
The multi-run plots show consistency across different random seeds, demonstrating the robustness of the learning algorithm. 