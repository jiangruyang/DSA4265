#!/usr/bin/env python3
"""
Launcher script for the Credit Card Rewards Optimizer Streamlit app.
This script sets up the correct Python path for imports before running Streamlit.
"""

import os
import sys
import subprocess

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # We're already at the project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variables to control Streamlit behavior
os.environ["STREAMLIT_WATCH_MODULES"] = "true"  # Enable module watching for dev experience
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings from HuggingFace tokenizers
os.environ["STREAMLIT_SERVER_WATCH_EXCLUDE_PATTERNS"] = "torch.*,torchvision.*"  # Exclude PyTorch modules from watching

def main():
    """Run the Streamlit app with the correct Python path."""
    # Launch the Streamlit app
    streamlit_app_path = os.path.join(project_root, "src", "user_interface", "Welcome.py")
    
    # Use subprocess to launch Streamlit with the proper Python path
    cmd = ["streamlit", "run", streamlit_app_path]
    print(f"Launching Streamlit app: {' '.join(cmd)}")
    
    # Execute the command and forward all output
    subprocess.run(cmd, env=os.environ, check=True)

if __name__ == "__main__":
    main() 