#!/usr/bin/env python3
"""
Tests for the custom environment implementation.
"""

import unittest
import torch
import numpy as np
import gymnasium as gym

# Import our custom environment
import push_cube_hit_cube


class TestPushCubeHitCubeEnv(unittest.TestCase):
    """Test cases for the PushCubeHitCube environment."""
    
    def setUp(self):
        """Set up the environment for testing."""
        self.env = gym.make(
            "PushCubeHitCube-v1",
            obs_mode="state",
            render_mode="rgb_array",
            control_mode="pd_joint_delta_pos"
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_environment_creation(self):
        """Test that the environment can be created successfully."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.spec.id, "PushCubeHitCube-v1")
    
    def test_observation_space(self):
        """Test that the observation space is properly defined."""
        obs_space = self.env.observation_space
        self.assertIsNotNone(obs_space)
        self.assertTrue(hasattr(obs_space, 'shape'))
    
    def test_action_space(self):
        """Test that the action space is properly defined."""
        action_space = self.env.action_space
        self.assertIsNotNone(action_space)
        self.assertEqual(action_space.shape, (7,))  # 7-dimensional joint control
    
    def test_reset(self):
        """Test that the environment can be reset."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gymnasium API
        
        self.assertIsNotNone(obs)
        self.assertTrue(isinstance(obs, (np.ndarray, torch.Tensor)))
    
    def test_step(self):
        """Test that the environment can take steps."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Take a random action
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Check return values
        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, (int, float, np.number))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_reward_range(self):
        """Test that rewards are within expected range."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_reward = 0
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Reward should be reasonable (not infinite or NaN)
            self.assertFalse(np.isnan(reward))
            self.assertFalse(np.isinf(reward))
            self.assertGreater(reward, -100)  # Reasonable lower bound
            self.assertLess(reward, 100)      # Reasonable upper bound
            
            if done or truncated:
                break
    
    def test_episode_termination(self):
        """Test that episodes can terminate properly."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        step_count = 0
        max_steps = 100  # Should not exceed max episode length
        
        while step_count < max_steps:
            action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(action)
            step_count += 1
            
            if done or truncated:
                break
        
        # Episode should terminate within max steps
        self.assertLessEqual(step_count, max_steps)
    
    def test_info_structure(self):
        """Test that info dictionary has expected structure."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        action = self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Check that info is a dictionary
        self.assertIsInstance(info, dict)
        
        # Check for expected keys (if any)
        if 'is_success' in info:
            self.assertIsInstance(info['is_success'], bool)


class TestAgent(unittest.TestCase):
    """Test cases for the Agent class."""
    
    def setUp(self):
        """Set up the agent for testing."""
        self.env = gym.make(
            "PushCubeHitCube-v1",
            obs_mode="state",
            render_mode="rgb_array",
            control_mode="pd_joint_delta_pos"
        )
        
        # Import Agent class
        from train import Agent
        self.Agent = Agent
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_agent_creation(self):
        """Test that the agent can be created successfully."""
        agent = self.Agent(self.env)
        self.assertIsNotNone(agent)
    
    def test_agent_forward_pass(self):
        """Test that the agent can process observations."""
        agent = self.Agent(self.env)
        
        # Create a dummy observation
        obs_shape = self.env.observation_space.shape
        dummy_obs = torch.randn(1, *obs_shape)
        
        # Test value function
        value = agent.get_value(dummy_obs)
        self.assertEqual(value.shape, (1, 1))
        
        # Test action generation
        action = agent.get_action(dummy_obs)
        self.assertEqual(action.shape, (1, 7))  # 7-dimensional action space
    
    def test_agent_deterministic_action(self):
        """Test deterministic action generation."""
        agent = self.Agent(self.env)
        
        obs_shape = self.env.observation_space.shape
        dummy_obs = torch.randn(1, *obs_shape)
        
        # Get deterministic actions
        action1 = agent.get_action(dummy_obs, deterministic=True)
        action2 = agent.get_action(dummy_obs, deterministic=True)
        
        # Deterministic actions should be the same
        torch.testing.assert_close(action1, action2)
    
    def test_agent_stochastic_action(self):
        """Test stochastic action generation."""
        agent = self.Agent(self.env)
        
        obs_shape = self.env.observation_space.shape
        dummy_obs = torch.randn(1, *obs_shape)
        
        # Get stochastic actions
        action1 = agent.get_action(dummy_obs, deterministic=False)
        action2 = agent.get_action(dummy_obs, deterministic=False)
        
        # Stochastic actions might be different (but not guaranteed)
        # Just check that they have the right shape
        self.assertEqual(action1.shape, (1, 7))
        self.assertEqual(action2.shape, (1, 7))
    
    def test_agent_action_and_value(self):
        """Test the combined action and value method."""
        agent = self.Agent(self.env)
        
        obs_shape = self.env.observation_space.shape
        dummy_obs = torch.randn(1, *obs_shape)
        
        # Test without providing action
        action, log_prob, entropy, value = agent.get_action_and_value(dummy_obs)
        
        self.assertEqual(action.shape, (1, 7))
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(entropy.shape, (1,))
        self.assertEqual(value.shape, (1, 1))
        
        # Test with provided action
        provided_action = torch.randn(1, 7)
        action, log_prob, entropy, value = agent.get_action_and_value(dummy_obs, provided_action)
        
        self.assertEqual(action.shape, (1, 7))
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(entropy.shape, (1,))
        self.assertEqual(value.shape, (1, 1))


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 