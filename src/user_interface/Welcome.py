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
from src.user_interface.components import page_header, section_header, nav_buttons

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

# Main app header with standardized component
page_header(
    title="Credit Card Rewards Optimizer: Singapore Edition 🇸🇬",
    icon="💳",
    description="Welcome to the Credit Card Rewards Optimizer for Singapore! Navigate through the pages to set up your profile, enter your spending data, and get personalized credit card recommendations."
)

# App workflow section
section_header(
    title="How It Works",
    description="Complete these steps to get personalized credit card recommendations:"
)

# Create a step-by-step guide using columns with icons
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 1️⃣ User Profile")
    st.markdown("Set your preferences and financial information")
    st.page_link("pages/1_User_Profile.py", label="Start Here", icon="👤", use_container_width=True)

with col2:
    st.markdown("### 2️⃣ Spending Input")
    st.markdown("Enter your spending data manually or upload statements")
    if st.session_state.preferences:
        st.page_link("pages/2_Spending_Input.py", label="Next Step", icon="💰", use_container_width=True)
    else:
        st.button("Complete Profile First", disabled=True, use_container_width=True)

with col3:
    st.markdown("### 3️⃣ Recommendations")
    st.markdown("Get personalized credit card recommendations")
    if st.session_state.spending_profile:
        st.page_link("pages/3_Recommendations.py", label="View Recommendations", icon="💳", use_container_width=True)
    else:
        st.button("Enter Spending First", disabled=True, use_container_width=True)

with col4:
    st.markdown("### 4️⃣ Chat with AI")
    st.markdown("Ask questions about card options and scenarios")
    if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
        st.page_link("pages/4_Chat.py", label="Chat with AI", icon="🤖", use_container_width=True)
    else:
        st.button("Get Recommendations First", disabled=True, use_container_width=True)

# Information about the app navigation
st.info("Use the sidebar to navigate between different sections of the app.")

# Footer with divider instead of markdown
st.divider()
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
