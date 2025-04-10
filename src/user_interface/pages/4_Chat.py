import streamlit as st
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Set page configuration
st.set_page_config(
    page_title="AI Chat | Credit Card Optimizer",
    page_icon="🤖",
)

# Import run_async from main app
from src.user_interface.streamlit_app import run_async

st.title("Chat with AI Assistant")
st.header("Ask About Your Cards & Explore Scenarios")

# Variables for consistent reference
client = st.session_state.client
agent = st.session_state.agent

if not st.session_state.recommendations:
    st.warning("Please generate recommendations first before using the chat feature.")
    st.info("Go to the Recommendations page to generate card recommendations.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")
    
    # Suggested questions
    st.subheader("Suggested Questions")
    suggested_questions = [
        "What is the annual fee?",
        "Do miles expire?",
        "What if I double my dining expenses?",
        "What if I travel overseas more frequently?",
        "Can I get lounge access with these cards?"
    ]
    
    selected_question = st.selectbox("Select a question or type your own below:", 
                                     [""] + suggested_questions)
    
    # Chat input
    user_question = st.text_input("Ask a question:", value=selected_question)
    
    if st.button("Send") and user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner("Thinking..."):
            # Check if it's a scenario question
            if "what if" in user_question.lower() or "double" in user_question.lower() or "more" in user_question.lower():
                response = run_async(
                    agent.analyze_scenario,
                    current_spending=st.session_state.spending_profile,
                    scenario_change=user_question,
                    current_recommendation=st.session_state.recommendations
                )
                
                answer = f"Recommendation: {response['recommendation_change']}\n"
                answer += f"Value Change: ${response['value_change']:.2f}\n\n"
                answer += "Updated Strategy:\n"
                for card_id, strategies in response['updated_strategy'].items():
                    card_name = next((c['name'] for c in st.session_state.recommendations['top_cards'] if c['id'] == card_id), card_id)
                    answer += f"{card_name}:\n"
                    for strategy in strategies:
                        answer += f"- {strategy['category'].capitalize()}: {strategy['reason']}\n"
            
            # Otherwise treat as T&C question
            else:
                # Use the first recommended card as default
                card_id = st.session_state.recommendations['top_cards'][0]['id']
                tc_response = run_async(
                    agent.query_terms_and_conditions,
                    user_question,
                    card_id
                )
                answer = f"{tc_response['answer']}\n\nSource: {tc_response['source']}"
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.experimental_rerun()

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.page_link("pages/3_Recommendations.py", label="← Recommendations", icon="💳")
with col2:
    st.page_link("streamlit_app.py", label="Home", icon="🏠") 