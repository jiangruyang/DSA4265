import streamlit as st
import os
import sys
import asyncio
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.mcp.client import CardOptimizerClient
from src.agent.agent import CardOptimizerAgent
from src.statement_processing.merchant_categorizer import MerchantCategorizer

# Initialize session state
if "preferences" not in st.session_state:
    st.session_state.preferences = {}
if "spending_profile" not in st.session_state:
    st.session_state.spending_profile = {}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "client" not in st.session_state:
    st.session_state.client = CardOptimizerClient()
if "agent" not in st.session_state:
    st.session_state.agent = CardOptimizerAgent()

# Initialize components
client = st.session_state.client
agent = st.session_state.agent
categorizer = MerchantCategorizer()

# Helper function to run async functions
def run_async(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(func(*args, **kwargs))
    loop.close()
    return result

# Initialize client if needed
if not client.initialized:
    with st.spinner("Initializing client..."):
        run_async(client.initialize)

# Page title
st.title("Credit Card Rewards Optimizer: Singapore Edition")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["User Profile", "Spending Input", "Recommendations", "Chat"])

# User Profile Page
if page == "User Profile":
    st.header("User Preferences")
    
    with st.form("preferences_form"):
        reward_type = st.selectbox("Reward Type", ["miles", "cashback", "points", "no preference"])
        annual_fee = st.slider("Maximum Annual Fee (SGD)", 0, 1000, 200)
        income = st.number_input("Annual Income (SGD)", min_value=0, value=60000)
        airport_lounge = st.checkbox("Prefer Airport Lounge Access")
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        citizenship = st.selectbox("Citizenship Status", ["Singaporean", "PR", "Foreigner"])
        additional_info = st.text_area("Additional Information (Optional)")
        
        submit_button = st.form_submit_button("Save Preferences")
        
        if submit_button:
            st.session_state.preferences = {
                'reward_type': reward_type,
                'max_annual_fee': annual_fee,
                'min_income': income,
                'prefer_airport_lounge': airport_lounge,
                'gender': gender,
                'citizenship': citizenship,
                'additional_info': additional_info
            }
            st.success("Preferences saved! Please proceed to Spending Input.")

# Spending Input Page
elif page == "Spending Input":
    st.header("Spending Information")
    
    input_method = st.radio("Input Method", ["Upload Statements", "Manual Entry"])
    
    if input_method == "Upload Statements":
        st.subheader("Upload Credit Card Statements (PDF)")
        uploaded_file = st.file_uploader("Choose a file", type="pdf")
        
        if uploaded_file is not None:
            # Save file temporarily
            with open(os.path.join("data", "card_tcs", "pdf", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File {uploaded_file.name} uploaded successfully! (PDF parsing to be implemented)")
            
            # Placeholder for PDF parsing
            st.info("PDF parsing would happen here in the actual implementation")
            
            # Dummy transactions for demo
            transactions = [
                {'merchant': 'NTUC FairPrice', 'amount': 200.50, 'date': '2023-08-01'},
                {'merchant': 'Grab Transport', 'amount': 150.75, 'date': '2023-08-03'},
                {'merchant': 'McDonald\'s', 'amount': 25.60, 'date': '2023-08-05'},
                {'merchant': 'Uniqlo Somerset', 'amount': 120.00, 'date': '2023-08-07'},
                {'merchant': 'Netflix Subscription', 'amount': 19.90, 'date': '2023-08-10'},
            ]
            
            # Process with categorizer
            st.session_state.spending_profile = categorizer.process_transactions(transactions)
            
            # Display categorized spending
            st.subheader("Categorized Spending")
            for category, amount in sorted(st.session_state.spending_profile.items(), key=lambda x: x[1], reverse=True):
                if amount > 0:
                    st.write(f"- {category.capitalize()}: ${amount:.2f}")
    
    else:  # Manual Entry
        st.subheader("Enter Monthly Spending by Category")
        
        with st.form("manual_spending_form"):
            # Get categories from categorizer
            categories = categorizer.get_categories()
            
            # Create input fields for each category
            spending_inputs = {}
            for category in categories:
                spending_inputs[category] = st.number_input(f"{category.capitalize()} (SGD)", min_value=0.0, value=0.0, step=10.0)
            
            submit_spending = st.form_submit_button("Save Spending Profile")
            
            if submit_spending:
                st.session_state.spending_profile = spending_inputs
                st.success("Spending profile saved! Please proceed to Recommendations.")

# Recommendations Page
elif page == "Recommendations":
    st.header("Credit Card Recommendations")
    
    if not st.session_state.preferences:
        st.warning("Please complete the User Profile first.")
    elif not st.session_state.spending_profile:
        st.warning("Please complete the Spending Input first.")
    else:
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing your spending profile and generating recommendations..."):
                # Call agent for recommendations
                recommendations = run_async(
                    agent.recommend_cards,
                    st.session_state.spending_profile,
                    st.session_state.preferences
                )
                st.session_state.recommendations = recommendations
        
        if st.session_state.recommendations:
            st.subheader("Top Recommended Cards")
            
            for i, card in enumerate(st.session_state.recommendations['top_cards'], 1):
                with st.expander(f"{i}. {card['name']} (Priority: {card['priority']})"):
                    # Card details would come from get_card_details
                    card_details = run_async(client.get_card_details, card['id'])
                    
                    st.write(f"**Annual Fee:** ${card_details.get('annual_fee', 0):.2f}")
                    st.write(f"**Promotion:** {card_details.get('promotion', 'N/A')}")
                    
                    if card['id'] in st.session_state.recommendations['usage_strategy']:
                        st.write("**Optimal Usage Strategy:**")
                        for usage in st.session_state.recommendations['usage_strategy'][card['id']]:
                            st.write(f"- {usage['category'].capitalize()}: {usage['reason']}")
            
            st.subheader("Estimated Monthly Value")
            st.write(f"${st.session_state.recommendations.get('estimated_value', 0):.2f}")
            
            st.info("Proceed to the Chat tab to ask questions about these recommendations or explore scenarios.")

# Chat Page
elif page == "Chat":
    st.header("Ask About Your Cards & Explore Scenarios")
    
    if not st.session_state.recommendations:
        st.warning("Please generate recommendations first before using the chat feature.")
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

# Ensure proper cleanup when the app exits
def on_shutdown():
    if st.session_state.client and st.session_state.client.initialized:
        run_async(st.session_state.client.shutdown)
    if st.session_state.agent and st.session_state.agent.initialized:
        run_async(st.session_state.agent.shutdown)

# Register shutdown handler
if hasattr(st, 'on_script_end'):
    st.on_script_end(on_shutdown)

# Run the app with: streamlit run src/app.py 