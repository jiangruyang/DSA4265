#!/usr/bin/env python3
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

# Create necessary directories
os.makedirs("data/vector_db", exist_ok=True)
os.makedirs("data/card_tcs/pdf", exist_ok=True)

print("Starting Credit Card Rewards Optimizer...")
print("If you have T&C PDFs, make sure they are in the data/card_tcs/pdf directory")

# Run Streamlit app
os.system("streamlit run src/user_interface/streamlit_app.py")