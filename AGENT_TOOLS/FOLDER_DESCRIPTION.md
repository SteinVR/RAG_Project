# AGENT_TOOLS Directory

A folder for storing agents' auxiliary scripts and tools. These scripts are used for development, testing, and demonstration purposes and are not involved in the core project logic.

## Contents

### Documentation
- `VISUAL_PROGRESS_FEATURE.md` - Documentation for the visual progress indicators feature implementation

### Demonstration Scripts
- `demo_progress.py` - Interactive demo of the ConsoleProgress utility showcasing all visual indicators

### Test Scripts
- `test_visual_progress.py` - Test script for verifying visual progress indicators in the full RAG pipeline (requires API keys)

## Usage Guidelines

Scripts in this folder:
- Should NOT be imported or used by the main application (`src/` modules)
- Are intended for development, testing, and documentation purposes
- May require additional setup or environment variables
- Are not part of the production deployment
