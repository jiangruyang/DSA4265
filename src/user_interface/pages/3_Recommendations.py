import streamlit as st
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Set page configuration
st.set_page_config(
    page_title="Recommendations | Credit Card Optimizer",
    page_icon="💳",
)

# Import run_async from main app
from src.user_interface.streamlit_app import run_async

st.title("Credit Card Recommendations")

# Variables for consistent reference
client = st.session_state.client
agent = st.session_state.agent

if not st.session_state.preferences:
    st.warning("Please complete the User Profile first.")
    st.info("Go to the User Profile page to set your preferences.")
elif not st.session_state.spending_profile:
    st.warning("Please complete the Spending Input first.")
    st.info("Go to the Spending Input page to enter your spending information.")
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

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.page_link("pages/2_Spending_Input.py", label="← Spending Input", icon="💰")
with col3:
    if "recommendations" in st.session_state and st.session_state.recommendations:
        st.page_link("pages/4_Chat.py", label="Next: Chat with AI →", icon="🤖")
    else:
        st.warning("Please generate recommendations before proceeding to chat.") 