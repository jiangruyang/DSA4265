import streamlit as st
import os
import sys

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utils for consistent event loop
from src.user_interface.utils import initialize_app_event_loop

# Set page configuration
st.set_page_config(
    page_title="User Profile | Credit Card Optimizer",
    page_icon="👤",
    layout="wide"
)

# Ensure the application event loop is initialized
initialize_app_event_loop()

st.title("👤 User Profile")
st.markdown("Please provide your information to help us recommend the best credit cards for your needs.")

with st.form("preferences_form"):
    # Section 1: Personal and Financial Profile
    st.markdown("## 1. Personal and Financial Profile")
    st.markdown("This information helps us identify cards you're eligible for.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.radio(
            "Gender",
            ["Male", "Female", "Prefer Not to Say"],
            index=["Male", "Female", "Prefer not to say"].index(st.session_state.preferences.get('gender', "Prefer not to say")) if 'gender' in st.session_state.preferences else 2
        )
    
    with col2:
        citizenship = st.radio(
            "Citizenship Status",
            ["Singaporean", "PR", "Foreigner"],
            index=["Singaporean", "PR", "Foreigner"].index(st.session_state.preferences.get('citizenship', "Singaporean")) if 'citizenship' in st.session_state.preferences else 0
        )
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input(
            "Annual Income (SGD)",
            min_value=0,
            step=1000,
            value=st.session_state.preferences.get('min_income', 60000),
            help="Net salary after CPF contributions/taxes"
        )
    
    with col2:
        debt_obligation = st.number_input(
            "Total Monthly Debt Obligation (SGD)",
            min_value=0,
            value=st.session_state.preferences.get('debt_obligation', 0),
            help="Housing loans, car loans, and other regular financial commitments"
        )
    
    st.markdown("")
    st.markdown("")
    
    # Section 2: Lifestyle & Preferences
    st.markdown("## 2. Lifestyle & Preferences")
    st.markdown("Tell us about your preferences to help us match cards to your lifestyle.")
    st.markdown("---")
    
    reward_type = st.radio(
        "Rewards Preference",
        ["Cashback", "Air Miles", "Both"],
        index=["cashback", "air miles", "both"].index(st.session_state.preferences.get('reward_type', "cashback")) if 'reward_type' in st.session_state.preferences and st.session_state.preferences['reward_type'] in ["cashback", "air miles", "both"] else 0,
        horizontal=True
    )
    
    st.markdown("")
    
    if reward_type in ["Air Miles", "Both"]:
        airlines_options = [
            "Singapore Airlines/Scoot",
            "Qantas/Jetstar",
            "AirAsia",
            "Emirates",
            "Qatar Airways",
            "Cathay Pacific",
            "Others",
            "No Preferred Airlines"
        ]
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            preferred_airline = st.selectbox(
                "Preferred Airlines",
                airlines_options,
                index=airlines_options.index(st.session_state.preferences.get('preferred_airline', "No preferred Airlines").title() if st.session_state.preferences.get('preferred_airline') == "No preferred Airlines" else st.session_state.preferences.get('preferred_airline', "No Preferred Airlines")) if 'preferred_airline' in st.session_state.preferences and st.session_state.preferences['preferred_airline'].title() in [opt.title() for opt in airlines_options] else 7
            )
        
        with col2:
            if preferred_airline == "Others":
                other_airline = st.text_input(
                    "Specify Your Preferred Airline",
                    value=st.session_state.preferences.get('other_airline', "")
                )
            else:
                other_airline = st.session_state.preferences.get('other_airline', "")
        
        st.markdown("")
    else:
        preferred_airline = "No Preferred Airlines"
        other_airline = ""
    
    annual_fee = st.slider(
        "Maximum Annual Fee (SGD)",
        0, 2000,
        st.session_state.preferences.get('max_annual_fee', 400),
        step=10,
        help="The highest annual fee you're willing to pay for a credit card"
    )
    
    st.markdown("")
    
    spending_goals = st.text_area(
        "Spending Goals & Priorities",
        value=st.session_state.preferences.get('spending_goals', ""),
        placeholder="E.g., 'I want to save more on daily essentials,' 'I travel frequently and want lounge access'"
    )
    
    st.markdown("")
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    with col2:
        submit_button = st.form_submit_button("💾 Save Preferences", use_container_width=True)
    
    if submit_button:
        # Convert display values back to stored values for consistency
        reward_type_map = {"Cashback": "cashback", "Air Miles": "air miles", "Both": "both"}
        
        st.session_state.preferences = {
            'gender': gender if gender != "Prefer Not to Say" else "Prefer not to say",
            'citizenship': citizenship,
            'min_income': income,
            'debt_obligation': debt_obligation,
            'reward_type': reward_type_map.get(reward_type, reward_type.lower()),
            'preferred_airline': preferred_airline if preferred_airline != "No Preferred Airlines" else "No preferred Airlines",
            'other_airline': other_airline,
            'max_annual_fee': annual_fee,
            'spending_goals': spending_goals
        }
        st.success("✅ Preferences Saved! Please Proceed to Spending Input.")

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.page_link("streamlit_app.py", label="← Home", icon="🏠")
with col3:
    if st.session_state.preferences:
        st.page_link("pages/2_Spending_Input.py", label="Next: Spending Input →", icon="💰")
    else:
        st.warning("⚠️ Please Save Your Preferences Before Proceeding.") 