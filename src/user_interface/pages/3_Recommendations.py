import streamlit as st
import os
import sys

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required components
from src.agent.agent import CardOptimizerAgent
from src.user_interface.utils import run_async, initialize_app_event_loop

# Set page configuration
st.set_page_config(
    page_title="Recommendations | Credit Card Optimizer",
    page_icon="💳",
)

# Ensure the application event loop is initialized
initialize_app_event_loop()

st.title("Credit Card Recommendations")

# Initialize agent if not already in session state
if "agent" not in st.session_state:
    st.session_state.agent = CardOptimizerAgent()
    # Initialize the agent (agent will handle client initialization)
    with st.spinner("Initializing agent..."):
        run_async(st.session_state.agent.initialize)

# Initialize chat_history if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Variable for consistent reference
agent = st.session_state.agent

if not st.session_state.preferences:
    st.warning("Please complete the User Profile first.")
    st.info("Go to the User Profile page to set your preferences.")
elif not st.session_state.spending_profile:
    st.warning("Please complete the Spending Input first.")
    st.info("Go to the Spending Input page to enter your spending information.")
else:
    if not st.session_state.chat_history or len(st.session_state.chat_history) == 0:
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing your spending profile and generating recommendations..."):
                # Format the initial message for recommendations
                initial_message = "I would like credit card recommendations based on my spending profile and preferences."
                
                # Prepare context with user data
                context = {
                    'spending_profile': st.session_state.spending_profile,
                    'preferences': st.session_state.preferences
                }
                
                # Get recommendation from agent
                try:
                    # Using the shared application event loop via run_async
                    recommendation = run_async(
                        agent.send_message,
                        initial_message,
                        context
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": initial_message})
                    st.session_state.chat_history.append({"role": "assistant", "content": recommendation})
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.session_state.chat_history.append({"role": "user", "content": initial_message})
                    st.session_state.chat_history.append({"role": "assistant", "content": f"I'm sorry, I encountered an error while generating recommendations: {str(e)}. Please try again later."})
    
    # Display recommendation if available
    if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
        st.markdown("## Your Recommendation")
        st.markdown(st.session_state.chat_history[1]["content"])
        
        # Add info about chatting with the agent
        st.info("Proceed to the Chat tab to ask questions about these recommendations or explore scenarios.")

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.page_link("pages/2_Spending_Input.py", label="← Spending Input", icon="💰")
with col3:
    if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
        st.page_link("pages/4_Chat.py", label="Next: Chat with AI →", icon="🤖")
    else:
        st.warning("Please generate recommendations before proceeding to chat.") 