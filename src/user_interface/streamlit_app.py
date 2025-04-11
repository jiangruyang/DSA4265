import os
# Set tokenizers parallelism to false to avoid deadlocks with forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import sys
import asyncio
import atexit
import threading
from typing import Dict, List, Any

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from utils - use absolute import from project root
from src.user_interface.utils import run_async, initialize_app_event_loop

# Set page configuration
st.set_page_config(
    page_title="Credit Card Rewards Optimizer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the application event loop early
initialize_app_event_loop()

from src.statement_processing.merchant_categorizer import MerchantCategorizer
# Import these only when needed in specific pages
# from src.mcp.client import CardOptimizerClient
# from src.agent.agent import CardOptimizerAgent

# Initialize session state for shared data across pages
if "preferences" not in st.session_state:
    st.session_state.preferences = {}
if "categorizer" not in st.session_state:
    st.session_state.categorizer = MerchantCategorizer()
if "spending_profile" not in st.session_state:
    st.session_state.spending_profile = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# client and agent initialization moved to Recommendations page
# if "client" not in st.session_state:
#     st.session_state.client = CardOptimizerClient()
# if "agent" not in st.session_state:
#     st.session_state.agent = CardOptimizerAgent()

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
    if "app_loop" in st.session_state and st.session_state.app_loop:
        # Stop the event loop gracefully
        if not st.session_state.app_loop.is_closed():
            # Schedule all shutdown tasks in the loop
            if "client" in st.session_state and st.session_state.client:
                asyncio.run_coroutine_threadsafe(
                    st.session_state.client.shutdown(), 
                    st.session_state.app_loop
                )
            if "agent" in st.session_state and st.session_state.agent:
                asyncio.run_coroutine_threadsafe(
                    st.session_state.agent.shutdown(), 
                    st.session_state.app_loop
                )
            
            # Give time for tasks to complete and stop the loop
            def shutdown_loop():
                st.session_state.app_loop.call_soon_threadsafe(
                    st.session_state.app_loop.stop
                )
            
            # Schedule shutdown after 2 seconds to allow tasks to complete
            shutdown_timer = threading.Timer(2.0, shutdown_loop)
            shutdown_timer.start()

# Register the shutdown handler
atexit.register(on_shutdown)

# Run the app with: streamlit run src/app.py 