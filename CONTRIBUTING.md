# Contributing to Robotic Manipulation RL

Thank you for your interest in contributing to the Robotic Manipulation RL project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with code changes
- **Documentation**: Improve or add documentation
- **Examples**: Add new examples or tutorials
- **Tests**: Add or improve test coverage

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/WeiHanTu/robotic-manipulation-rl.git
   cd robotic-manipulation-rl
   ```

2. **Set up the development environment**
   ```bash
   make setup
   # or manually:
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🧪 Development Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ManiSkill environment

### Installation

```bash
# Clone the repository
git clone https://github.com/WeiHanTu/robotic-manipulation-rl.git
cd robotic-manipulation-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or manually:
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_environment.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run both
make check
```

## 📝 Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 127 characters
- **Indentation**: 4 spaces
- **Import organization**: Standard library, third-party, local imports

### Code Formatting

We use `black` for code formatting:

```bash
black . --line-length 127
```

### Linting

We use `flake8` for linting:

```bash
flake8 . --max-line-length=127 --max-complexity=10
```

### Type Hints

We encourage the use of type hints for better code documentation:

```python
from typing import Optional, Tuple, Dict, Any

def process_observation(obs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Process observation tensor."""
    return obs.to(device)
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

### Test Structure

```python
import unittest
import torch
import gymnasium as gym

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.env = gym.make("PushCubeHitCube-v1")
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_environment_creation(self):
        """Test that environment can be created."""
        self.assertIsNotNone(self.env)
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test
python -m pytest tests/test_environment.py::TestEnvironment::test_environment_creation

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

## 📚 Documentation Guidelines

### Docstrings

Use Google-style docstrings:

```python
def train_agent(env_id: str, total_steps: int) -> Dict[str, Any]:
    """Train an agent on the specified environment.
    
    Args:
        env_id: The environment ID to train on.
        total_steps: Total number of training steps.
        
    Returns:
        Dictionary containing training metrics.
        
    Raises:
        ValueError: If env_id is not supported.
    """
    pass
```

### README Updates

When adding new features, update the README.md to include:

- Installation instructions for new dependencies
- Usage examples
- Configuration options
- Performance benchmarks

## 🚀 Submitting Changes

### Pull Request Process

1. **Ensure your code passes tests**
   ```bash
   make test
   make check
   ```

2. **Update documentation**
   - Update README.md if needed
   - Add docstrings for new functions
   - Update API documentation

3. **Create a pull request**
   - Use a descriptive title
   - Include a detailed description
   - Reference any related issues

### Pull Request Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated existing tests if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 1.9.0]
- CUDA version: [e.g., 11.1]

## Additional Information
Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature.

## Motivation
Why this feature would be useful.

## Proposed Implementation
Optional: How you think this could be implemented.

## Alternatives Considered
Optional: Other approaches you considered.

## Additional Context
Any other relevant information.
```

## 🏷️ Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release tag
- [ ] Publish to PyPI (if applicable)

## 📞 Getting Help

### Communication Channels

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Pull Requests**: For code contributions

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and considerate in all interactions.

## 🙏 Acknowledgments

Thank you to all contributors who help improve this project!

---

For any questions about contributing, please open an issue or start a discussion on GitHub. 