# Makefile for Robotic Manipulation RL Project

.PHONY: help install test train demo clean setup

# Default target
help:
	@echo "🤖 Robotic Manipulation RL Project"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Setup the project (install + create dirs)"
	@echo "  make test       - Run tests"
	@echo "  make train      - Run quick training"
	@echo "  make train-full - Run full training (3M steps)"
	@echo "  make baseline   - Run baseline training"
	@echo "  make demo       - Run demo with random actions"
	@echo "  make clean      - Clean up generated files"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Setup project
setup: install
	@echo "🔧 Setting up project..."
	mkdir -p runs videos figures/png
	@echo "✅ Project setup complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

# Quick training
train:
	@echo "🚀 Starting quick training..."
	python scripts/run_training.py --mode quick

# Full training
train-full:
	@echo "🚀 Starting full training..."
	python scripts/run_training.py --mode full --timesteps 3000000

# Baseline training
baseline:
	@echo "🚀 Starting baseline training..."
	python scripts/run_training.py --mode baseline

# Demo
demo:
	@echo "🎮 Running demo..."
	python demo_random_actions.py

# Multi-seed training
multi-train:
	@echo "🎲 Running multi-seed training..."
	python scripts/run_training.py multi

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -rf runs/* videos/* figures/png/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cleanup complete!"

# Lint code
lint:
	@echo "🔍 Running linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	@echo "🎨 Formatting code..."
	black . --line-length 127

# Check code quality
check: lint format
	@echo "✅ Code quality check complete!"

# Install development dependencies
install-dev: install
	@echo "🔧 Installing development dependencies..."
	pip install -e .
	pip install black flake8 pytest mypy

# Create virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python -m venv venv
	@echo "✅ Virtual environment created!"
	@echo "💡 Activate it with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

# Run example
example:
	@echo "📚 Running example..."
	python examples/training_example.py

# Show project info
info:
	@echo "📊 Project Information"
	@echo "====================="
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "Project files:"
	@ls -la *.py *.md *.txt 2>/dev/null || true
	@echo ""
	@echo "Directories:"
	@ls -la */ 2>/dev/null || true

# Help for specific commands
train-help:
	@echo "🎯 Training Options:"
	@echo "  make train        - Quick training (500k steps)"
	@echo "  make train-full   - Full training (3M steps)"
	@echo "  make baseline     - Baseline training (PickCube)"
	@echo "  make multi-train  - Multi-seed training"
	@echo ""
	@echo "Manual training:"
	@echo "  python train.py --env_id PushCubeHitCube-v1 --total_timesteps 1000000"
	@echo "  python baseline.py --env_id PickCube-v1 --total_timesteps 1000000"

demo-help:
	@echo "🎮 Demo Options:"
	@echo "  make demo         - Run random actions demo"
	@echo "  make example      - Run training example"
	@echo ""
	@echo "Manual demo:"
	@echo "  python demo_random_actions.py"
	@echo "  python examples/training_example.py"

# Default target
.DEFAULT_GOAL := help 