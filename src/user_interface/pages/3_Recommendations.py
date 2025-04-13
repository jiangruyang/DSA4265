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
# Import standardized components
from src.user_interface.components import page_header, section_header, subsection_header, progress_tracker, nav_buttons

# Set page configuration
st.set_page_config(
    page_title="Recommendations | Credit Card Optimizer",
    page_icon="💳",
    layout="wide"
)

# Ensure the application event loop is initialized
initialize_app_event_loop()

# Display progress tracker (Recommendations is step 2)
progress_tracker(current_step=2)

# Page header
page_header(
    title="Credit Card Recommendations",
    icon="💳",
    description="Review your personalized credit card recommendations based on your profile and spending habits."
)

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

# Check if prerequisites are completed
if not st.session_state.preferences:
    st.warning("Please complete the User Profile first.")
    st.info("Go to the User Profile page to set your preferences.")
    # Add direct navigation button
    st.page_link("pages/1_User_Profile.py", label="Go to User Profile", icon="👤", use_container_width=False)
elif not st.session_state.spending_profile:
    st.warning("Please complete the Spending Input first.")
    st.info("Go to the Spending Input page to enter your spending information.")
    # Add direct navigation button
    st.page_link("pages/2_Spending_Input.py", label="Go to Spending Input", icon="💰", use_container_width=False)
else:
    # Show either the generate button or the recommendation
    if not st.session_state.chat_history or len(st.session_state.chat_history) == 0:
        section_header(
            title="Generate Your Recommendations",
            description="Click the button below to analyze your spending profile and generate personalized credit card recommendations."
        )
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate Recommendations", type="primary", use_container_width=True):
                with st.status("Analyzing your data...", expanded=True) as status:
                    # Step 1: Format the initial message
                    st.write("Preparing your spending profile...")
                    initial_message = "I would like credit card recommendations based on my spending profile and preferences."
                    
                    # Step 2: Prepare context with user data
                    st.write("Analyzing spending patterns...")
                    context = {
                        'spending_profile': st.session_state.spending_profile,
                        'preferences': st.session_state.preferences
                    }
                    
                    # Step 3: Get recommendation from agent
                    st.write("Generating personalized recommendations...")
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
                        status.update(label="Recommendations generated successfully!", state="complete")
                    except Exception as e:
                        status.update(label=f"Error: {str(e)}", state="error")
                        st.error(f"Error generating recommendations: {str(e)}")
                        st.session_state.chat_history.append({"role": "user", "content": initial_message})
                        st.session_state.chat_history.append({"role": "assistant", "content": f"I'm sorry, I encountered an error while generating recommendations: {str(e)}. Please try again later."})
                
                # Force rerun to display the recommendations
                st.rerun()
    
    # Display recommendation if available
    if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
        section_header(
            title="Your AI Recommendation is Ready"
        )
        
        # Display recommendation in a highlighted container
        with st.container():
            # Insert "\$" before all "$" in the recommendation
            formatted_recommendation = st.session_state.chat_history[1]["content"].replace("$", "\\$")
            st.markdown(formatted_recommendation)
        
        # Add info about chatting with the agent
        st.info("Proceed to the Chat page to ask questions about these recommendations or explore scenarios.")

# Navigation buttons
st.divider()
nav_buttons(
    prev_page="pages/2_Spending_Input.py", 
    next_page="pages/4_Chat.py", 
    next_condition=bool(st.session_state.chat_history and len(st.session_state.chat_history) >= 2),
    next_warning="Please generate recommendations before proceeding to chat."
) 