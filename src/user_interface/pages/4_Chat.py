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
)

# Import from utils
from src.user_interface.utils import run_async, initialize_app_event_loop

# Ensure the application event loop is initialized
initialize_app_event_loop()

st.title("Chat with AI Assistant")
st.header("Ask About Your Cards & Explore Scenarios")

# Check if agent is initialized
if "agent" not in st.session_state:
    st.warning("Please go to the Recommendations page first to initialize the system.")
    st.page_link("pages/3_Recommendations.py", label="Go to Recommendations", icon="💳")
    st.stop()

# Variable for consistent reference
agent = st.session_state.agent

# Check if recommendations chat has been started
if not st.session_state.chat_history or len(st.session_state.chat_history) < 2:
    st.warning("Please generate recommendations first before using the chat feature.")
    st.info("Go to the Recommendations page to generate card recommendations.")
    st.page_link("pages/3_Recommendations.py", label="Go to Recommendations", icon="💳")
else:
    # Create a container for all chat history messages
    chat_history_container = st.container()
    
    # Display chat history within the container
    with chat_history_container:
        for message in st.session_state.chat_history[1:]:
            if message["role"] == "user":
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
    
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
                        
                        # Write answer 
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_message = f"I'm sorry, I encountered an error while processing your question: {str(e)}. Please try again."
                        st.error(error_message)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Rerun the app to update the chat display with the new messages
        st.rerun()
    
    # Suggested questions - place before footer
    st.subheader("Suggested Questions")
    suggested_questions = [
        "What is the annual fee?",
        "Do miles expire?",
        "What if I double my dining expenses?",
        "What if I travel overseas more frequently?",
        "Can I get lounge access with these cards?"
    ]
    
    # Create buttons for suggested questions
    cols = st.columns(len(suggested_questions))
    for i, question in enumerate(suggested_questions):
        if cols[i].button(question, key=f"question_{i}"):
            # Store the selected question in session state
            st.session_state.selected_question = question
            # Rerun immediately to process the question in the main flow
            st.rerun()

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.page_link("pages/3_Recommendations.py", label="← Recommendations", icon="💳")
    with col2:
        st.page_link("streamlit_app.py", label="Home", icon="🏠") 