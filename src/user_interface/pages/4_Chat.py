import streamlit as st
import os
import sys

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set page configuration
st.set_page_config(
    page_title="AI Chat | Credit Card Optimizer",
    page_icon="🤖",
    layout="wide"
)

# Import from utils
from src.user_interface.utils import run_async, initialize_app_event_loop
# Import standardized components
from src.user_interface.components import page_header, section_header, subsection_header, progress_tracker, nav_buttons

# Ensure the application event loop is initialized
initialize_app_event_loop()

# Display progress tracker (Chat is step 3)
progress_tracker(current_step=3)

# Initialize session state variables for suggested questions
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
    
if "refresh_questions" not in st.session_state:
    st.session_state.refresh_questions = True

# Page header
page_header(
    title="Chat with AI Assistant",
    icon="🤖",
    description="Ask questions about your card recommendations and explore different spending scenarios."
)

# Check if agent is initialized
if "agent" not in st.session_state:
    st.warning("Please go to the Recommendations page first to initialize the system.")
    st.page_link("pages/3_Recommendations.py", label="Go to Recommendations", icon="💳", use_container_width=False)
    st.stop()

# Variable for consistent reference
agent = st.session_state.agent

# Check if recommendations chat has been started
if not st.session_state.chat_history or len(st.session_state.chat_history) < 2:
    st.warning("Please generate recommendations first before using the chat feature.")
    st.info("Go to the Recommendations page to generate card recommendations.")
    st.page_link("pages/3_Recommendations.py", label="Go to Recommendations", icon="💳", use_container_width=False)
else:
    # Create a container for all chat history messages
    chat_history_container = st.container()
    
    # Display chat history within the container
    with chat_history_container:
        for message in st.session_state.chat_history[1:]:
            # Insert "\$" before all "$" in the message content
            formatted_message = message['content'].replace("$", "\\$")
            if message["role"] == "user":
                st.chat_message("user").write(formatted_message)
            else:
                st.chat_message("assistant").write(formatted_message)
    
    # Check if we have a pending question from a button click
    if "selected_question" in st.session_state and st.session_state.selected_question:
        user_question = st.session_state.selected_question
        st.session_state.selected_question = None  # Clear it to prevent reprocessing
        
        # Display user message in the chat history container
        with chat_history_container:
            st.chat_message("user").write(user_question)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Generate and display assistant response
        with chat_history_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Using the shared application event loop via run_async
                        answer = run_async(
                            agent.send_message,
                            user_question
                        )
                        
                        # Write answer 
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Set flag to refresh suggested questions after receiving an answer
                        st.session_state.refresh_questions = True
                    except Exception as e:
                        error_message = f"I'm sorry, I encountered an error while processing your question: {str(e)}. Please try again."
                        st.error(error_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Rerun is not needed here as we already handled the question
    
    # Create a container for chat input (will appear after all messages)
    chat_input_container = st.container()
    with chat_input_container:
        user_question = st.chat_input("Ask a question:")
    
    if user_question:
        # Display user message in the chat history container
        with chat_history_container:
            st.chat_message("user").write(user_question)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Generate and display assistant response
        with chat_history_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Using the shared application event loop via run_async
                        answer = run_async(
                            agent.send_message,
                            user_question
                        )

                        # Insert "\" before all "$" in the answer
                        answer = answer.replace("$", "\\$")
                        
                        # Write answer 
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Set flag to refresh suggested questions after receiving an answer
                        st.session_state.refresh_questions = True
                    except Exception as e:
                        error_message = f"I'm sorry, I encountered an error while processing your question: {str(e)}. Please try again."
                        st.error(error_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Rerun the app to update the chat display with the new messages
        st.rerun()
    
    # Suggested questions section
    st.markdown("---")
    st.subheader("Suggested Questions")
    st.caption("These personalized questions are based on your conversation and may help you get more value from your credit cards.")
    
    # Generate dynamic suggested questions based on conversation history
    if "suggested_questions" not in st.session_state or st.session_state.get("refresh_questions", False):
        with st.spinner("Generating suggested questions..."):
            try:
                # Get dynamic questions from the agent
                suggested_questions = run_async(
                    agent.generate_suggested_questions,
                    3  # Limit to 3 questions
                )
                st.session_state.suggested_questions = suggested_questions
                st.session_state.refresh_questions = False
            except Exception as e:
                st.error(f"Error generating suggested questions: {str(e)}")
                # Fallback to default questions if there's an error
                suggested_questions = [
                    "What is the annual fee?",
                    "Do miles expire?",
                    "What if I travel more frequently?"
                ]
                st.session_state.suggested_questions = suggested_questions
    else:
        # Use cached questions to avoid regenerating on every rerun
        suggested_questions = st.session_state.suggested_questions
    
    # Create a container for the suggested questions
    question_container = st.container()
    with question_container:
        # Use columns for better spacing
        cols = st.columns(len(suggested_questions))
        for i, question in enumerate(suggested_questions):
            with cols[i]:
                # Create a visually distinct container using Streamlit's native components
                st.markdown(f"**{question}**")
                # Add a button below the text
                if st.button(f"Ask this 💬", key=f"question_{i}", use_container_width=True):
                    # Store the selected question in session state
                    st.session_state.selected_question = question
                    # Set flag to refresh questions on next run (after this question is answered)
                    st.session_state.refresh_questions = True
                    # Rerun immediately to process the question in the main flow
                    st.rerun()

    # Add a refresh button for suggested questions
    if st.button("Refresh Suggestions 🔄", use_container_width=True):
        st.session_state.refresh_questions = True
        st.rerun()

# Navigation buttons
st.divider()
nav_buttons(
    prev_page="pages/3_Recommendations.py", 
    next_page=None, 
    home=True
) 