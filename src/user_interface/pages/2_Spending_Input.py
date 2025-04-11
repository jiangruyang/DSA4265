import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Spending Input | Credit Card Optimizer",
    page_icon="💰",
)

import pandas as pd
import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import run_async from utils and statement parser
from src.user_interface.utils import run_async
from src.statement_processing.pdf_statement_parser import PDFStatementParser
from src.statement_processing.merchant_categorizer import MerchantCategorizer

# Function to load the merchant categorizer (only used if not already in session state)
@st.cache_resource
def load_merchant_categorizer():
    try:
        model_path = "models/merchant_categorizer"
        categorizer = MerchantCategorizer(model_path=model_path)
        # Set the model to evaluation mode for inference
        if hasattr(categorizer, 'model') and categorizer.model is not None:
            categorizer.model.eval()
        return categorizer
    except Exception as e:
        st.error(f"Error initializing merchant categorizer: {str(e)}")
        return None

# Initialize session state for merchant categorizer if not already present
if 'merchant_categorizer' not in st.session_state or st.session_state.merchant_categorizer is None:
    st.session_state.merchant_categorizer = load_merchant_categorizer()

st.title("Spending Input")
st.header("Spending Information")

# Get merchant categorizer from session state
merchant_categorizer = st.session_state.merchant_categorizer

# Create a statement parser that uses the merchant categorizer
statement_parser = PDFStatementParser()
if merchant_categorizer:
    statement_parser.merchant_categorizer = merchant_categorizer

input_method = st.radio("Input Method", ["Upload Statements", "Manual Entry"])

# Initialize uploaded_file to None
uploaded_file = None

if input_method == "Upload Statements":
    st.subheader("Upload Credit Card Statements (PDF)")
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        # Process PDF directly in memory without saving to disk
        with st.spinner("Processing statement..."):
            try:
                # Use inference mode to avoid PyTorch file watcher issues
                with torch.inference_mode():
                    # Use the actual PDF parser implementation
                    transactions = statement_parser.parse_statement(uploaded_file, is_path=False)
                
                if not transactions:
                    st.warning("No transactions found in the statement. Please check if the file is a valid credit card statement or try manual entry.")
                else:
                    # Process with statement parser
                    st.session_state.spending_profile = statement_parser.process_transactions(transactions)
                    
                    # Show number of transactions processed
                    st.info(f"Successfully processed {len(transactions)} transactions from your statement.")
            except Exception as e:
                st.error(f"Error processing statement: {str(e)}")
                st.info("If you're having issues with statement upload, you can use manual entry instead.")
                transactions = []
        
        # Display categorized spending if transactions were found
        if 'spending_profile' in st.session_state and any(amount > 0 for amount in st.session_state.spending_profile.values()):
            st.subheader("Categorized Spending")
            spending_data = [
                {
                    "Category": category.capitalize(),
                    "Amount": f"${amount:.2f}"
                }
                for category, amount in sorted(
                    st.session_state.spending_profile.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                if amount > 0
            ]
            st.table(pd.DataFrame(spending_data))

else:  # Manual Entry
    st.subheader("Enter Monthly Spending by Category")
    
    with st.form("manual_spending_form"):
        # Get categories from categorizer
        categories = merchant_categorizer.get_categories() if merchant_categorizer else [
            "groceries", "dining", "transportation", "shopping", 
            "entertainment", "travel", "utilities", "healthcare", 
            "education", "others"
        ]
        
        # Create input fields for each category
        spending_inputs = {}
        for category in categories:
            spending_inputs[category] = st.number_input(f"{category.capitalize()} (SGD)", min_value=0.0, value=0.0, step=10.0)
        
        submit_spending = st.form_submit_button("Save Spending Profile")
        
        if submit_spending:
            st.session_state.spending_profile = spending_inputs
            st.success("Spending profile saved! Please proceed to Recommendations.")

# Check if form was submitted this run
form_submitted = 'submit_spending' in locals() and submit_spending

# Display current spending profile if it exists and if the user hasn't just uploaded a file or submitted the form
if 'spending_profile' in st.session_state and not (uploaded_file is not None or form_submitted):
    st.subheader("Current Spending Profile")
    spending_data = [
        {
            "Category": category.capitalize(),
            "Amount": f"${amount:.2f}"
        }
        for category, amount in sorted(
            st.session_state.spending_profile.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if amount > 0
    ]
    if spending_data:
        st.table(pd.DataFrame(spending_data))
    else:
        st.info("Your current spending profile has no spending in any category.")

# Navigation buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.page_link("pages/1_User_Profile.py", label="← User Profile", icon="👤")
with col3:
    if 'spending_profile' in st.session_state and st.session_state.spending_profile:
        st.page_link("pages/3_Recommendations.py", label="Next: Recommendations →", icon="💳")
    else:
        st.warning("Please save your spending profile before proceeding.") 