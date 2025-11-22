# SinceAI Project

This is a Python project that uses AI to interpret maritime horn signals according to COLREG (International Regulations for Preventing Collisions at Sea) rules.

## Project Management

This project uses **uv** - a modern Python project manager that's faster and more reliable than poetry or pip. UV handles dependency management, virtual environments, and project packaging.

## Getting Started

### Prerequisites

- Python 3.13+
- uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
# Clone and enter the project directory
cd /path/to/sinceai

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment (optional - uv run handles this automatically)
source .venv/bin/activate
```

## Common Commands

All Python commands should be run through `uv` to ensure they use the correct virtual environment:

### Running Scripts

```bash
# Generate training data
uv run make_training_data.py

# Run the main application
uv run main.py
```

### Development Commands

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv sync

# Run tests (if available)
uv run pytest

# Run with specific Python version
uv run --python 3.13 script.py
```

### Project Management

```bash
# Show project info
uv show

# List dependencies
uv pip list

# Check for outdated packages
uv pip list --outdated
```

## Project Structure

- `main.py` - Main application logic for processing horn sounds
- `make_training_data.py` - Script to generate training datasets
- `dataset/` - Training data organized by signal types (turn_port, turn_starboard, etc.)
- `signals/` - Input signal files
- `backgrounds/` - Background audio files
- `samples/` - Sample audio files
- `output/` - Generated output files

## About UV

UV is a next-generation Python package manager that provides:

- **Speed**: 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Simplicity**: Single tool for all Python project needs
- **Compatibility**: Drop-in replacement for pip/poetry workflows

Always use `uv run` instead of `python` to ensure commands run in the correct environment.
