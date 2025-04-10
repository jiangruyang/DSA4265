import streamlit as st
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Set page configuration
st.set_page_config(
    page_title="User Profile | Credit Card Optimizer",
    page_icon="👤",
)

st.title("User Profile")
st.header("User Preferences")

with st.form("preferences_form"):
    # Pre-fill form with existing preferences if they exist
    reward_type = st.selectbox(
        "Reward Type", 
        ["miles", "cashback", "points", "no preference"],
        index=["miles", "cashback", "points", "no preference"].index(st.session_state.preferences.get('reward_type', "no preference")) if 'reward_type' in st.session_state.preferences else 0
    )
    
    annual_fee = st.slider(
        "Maximum Annual Fee (SGD)", 
        0, 1000, 
        st.session_state.preferences.get('max_annual_fee', 200)
    )
    
    income = st.number_input(
        "Annual Income (SGD)", 
        min_value=0, 
        value=st.session_state.preferences.get('min_income', 60000)
    )
    
    airport_lounge = st.checkbox(
        "Prefer Airport Lounge Access",
        value=st.session_state.preferences.get('prefer_airport_lounge', False)
    )
    
    gender_options = ["Male", "Female", "Other", "Prefer not to say"]
    gender = st.selectbox(
        "Gender", 
        gender_options,
        index=gender_options.index(st.session_state.preferences.get('gender', "Prefer not to say")) if 'gender' in st.session_state.preferences else 3
    )
    
    citizenship_options = ["Singaporean", "PR", "Foreigner"]
    citizenship = st.selectbox(
        "Citizenship Status", 
        citizenship_options,
        index=citizenship_options.index(st.session_state.preferences.get('citizenship', "Singaporean")) if 'citizenship' in st.session_state.preferences else 0
    )
    
    additional_info = st.text_area(
        "Additional Information (Optional)",
        value=st.session_state.preferences.get('additional_info', "")
    )
    
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

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.page_link("streamlit_app.py", label="← Home", icon="🏠")
with col3:
    if st.session_state.preferences:
        st.page_link("pages/2_Spending_Input.py", label="Next: Spending Input →", icon="💰")
    else:
        st.warning("Please save your preferences before proceeding.") 