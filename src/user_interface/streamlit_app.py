import streamlit as st
import os
import sys
import asyncio
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import from utils
from src.user_interface.utils import run_async

# Set page configuration
st.set_page_config(
    page_title="Credit Card Rewards Optimizer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.statement_processing.merchant_categorizer import MerchantCategorizer
# from src.mcp.client import CardOptimizerClient
# from src.agent.agent import CardOptimizerAgent

# Initialize session state for shared data across pages
if "preferences" not in st.session_state:
    st.session_state.preferences = {}
if "categorizer" not in st.session_state:
    st.session_state.categorizer = MerchantCategorizer()
if "spending_profile" not in st.session_state:
    st.session_state.spending_profile = {}
# if "recommendations" not in st.session_state:
#     st.session_state.recommendations = {}
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "client" not in st.session_state:
#     st.session_state.client = CardOptimizerClient()
# if "agent" not in st.session_state:
#     st.session_state.agent = CardOptimizerAgent()

# Initialize client if needed
# if not st.session_state.client.initialized:
#     with st.spinner("Initializing client..."):
#         run_async(st.session_state.client.initialize)

# Main app header
st.title("Credit Card Rewards Optimizer: Singapore Edition")
st.write("Welcome to the Credit Card Rewards Optimizer for Singapore! Navigate through the pages to set up your profile, enter your spending data, and get personalized credit card recommendations.")

# Information about the app navigation
st.info("Use the sidebar to navigate between different sections of the app.")

# Footer
st.markdown("---")
st.caption("Developed for DSA4265 Project - Singapore Edition")

# Ensure proper cleanup when the app exits
def on_shutdown():
    if "client" in st.session_state and st.session_state.client:
        asyncio.run(st.session_state.client.shutdown())

# Run the app with: streamlit run src/app.py 