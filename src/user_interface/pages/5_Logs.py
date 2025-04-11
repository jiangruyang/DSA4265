import streamlit as st
import os
import sys
from pathlib import Path

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agent.agent import get_log_file_path, CardOptimizerAgent
from src.user_interface.utils import run_async, initialize_app_event_loop

# Set page configuration
st.set_page_config(
    page_title="Logs | Credit Card Optimizer",
    page_icon="📝",
)

# Ensure the application event loop is initialized
initialize_app_event_loop()

st.title("Agent Logs")
st.write("This page shows the logs from the Credit Card Optimizer Agent.")

# Initialize agent if not already in session state
if "agent" not in st.session_state:
    st.session_state.agent = CardOptimizerAgent()
    # Initialize the agent (agent will handle client initialization)
    with st.spinner("Initializing agent..."):
        run_async(st.session_state.agent.initialize)

# Get the log file path
try:
    log_file_path = get_log_file_path()
    log_file = Path(log_file_path)
    
    if log_file.exists():
        # File exists, get agent status to include log path in status
        with st.spinner("Checking agent status..."):
            try:
                status = run_async(st.session_state.agent.check_status)
                # Display agent status
                st.subheader("Agent Status")
                st.json(status)
            except Exception as e:
                st.error(f"Error checking agent status: {str(e)}")
        
        # Display the log file
        st.subheader("Log File Contents")
        st.write(f"Log file: `{log_file_path}`")
        
        # Add refresh button
        if st.button("Refresh Log"):
            st.rerun()
        
        # Read logs with pagination to handle large files
        log_size = log_file.stat().st_size
        if log_size > 1_000_000:  # 1MB
            st.warning(f"Log file is large ({log_size/1_000_000:.2f} MB). Showing the last 1000 lines.")
            with open(log_file, 'r') as f:
                # Read the last 1000 lines (approximately)
                lines = f.readlines()[-1000:]
                log_content = ''.join(lines)
        else:
            with open(log_file, 'r') as f:
                log_content = f.read()
        
        # Display log content in a text area
        st.text_area("Log Contents", log_content, height=500)
        
        # Option to download logs
        st.download_button(
            label="Download Full Log File",
            data=open(log_file, 'rb').read(),
            file_name=f"agent_log_{log_file.name}",
            mime="text/plain"
        )
    else:
        st.error(f"Log file not found at: {log_file_path}")
        st.write("The agent might not have generated any logs yet.")
except Exception as e:
    st.error(f"Error accessing log file: {str(e)}")
    st.write("Make sure the agent is properly initialized and has permission to write logs.")

# Separator
st.markdown("---")

# Navigation
col1, col2 = st.columns([1, 1])
with col1:
    st.page_link("pages/4_Chat.py", label="← Back to Chat", icon="💬") 