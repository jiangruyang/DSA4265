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
from datetime import datetime

# Fix import paths 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import run_async and event loop initialization from utils
from src.user_interface.utils import run_async, initialize_app_event_loop
from src.statement_processing.pdf_statement_parser import PDFStatementParser
from src.statement_processing.merchant_categorizer import MerchantCategorizer

# Ensure the application event loop is initialized
initialize_app_event_loop()

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

# Initialize session state variables if not already present
if 'merchant_categorizer' not in st.session_state or st.session_state.merchant_categorizer is None:
    st.session_state.merchant_categorizer = load_merchant_categorizer()

# Initialize form data in session state
if 'edited_transactions' not in st.session_state:
    st.session_state.edited_transactions = []

if 'add_mode' not in st.session_state:
    st.session_state.add_mode = False
    
# Initialize new transaction fields if not present
if 'new_merchant' not in st.session_state:
    st.session_state.new_merchant = ""
if 'new_amount' not in st.session_state:
    st.session_state.new_amount = 0.0
if 'new_category' not in st.session_state:
    st.session_state.new_category = "others"

# Function to add a new transaction
def add_transaction():
    merchant = st.session_state.new_merchant
    amount = st.session_state.new_amount
    category = st.session_state.new_category
    
    if merchant and amount > 0:
        st.session_state.edited_transactions.append({
            "merchant": merchant,
            "amount": amount,
            "category": category,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "include": True,
            "type": "withdrawal"
        })
        
        # Reset input fields by updating session state
        st.session_state.new_merchant = ""
        st.session_state.new_amount = 0.0
        # Keep category as is - typically users might add multiple items in same category
        st.session_state.new_category = category
        
        # Mark profile as needing update
        st.session_state.profile_saved = False

# Callback to generate spending profile from edited transactions
def generate_spending_profile():
    if 'edited_transactions' in st.session_state:
        # Filter transactions that are marked to include
        filtered_transactions = [
            {
                "merchant": t["merchant"],
                "amount": t["amount"],
                "category": t["category"],
                "date": t["date"],
                "type": t["type"]
            }
            for t in st.session_state.edited_transactions
            if t["include"]
        ]
        
        # Get parser and process transactions
        parser = PDFStatementParser()
        if 'merchant_categorizer' in st.session_state and st.session_state.merchant_categorizer:
            parser.merchant_categorizer = st.session_state.merchant_categorizer
            
        st.session_state.spending_profile = parser.process_transactions(filtered_transactions)
        st.session_state.profile_saved = True
        
        # Show success message
        st.session_state.show_success = True
        
        # Force a rerun to show the spending profile immediately
        st.rerun()

# User Interface
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
file_processed = False

if input_method == "Upload Statements":
    st.subheader("Upload Credit Card Statements (PDF)")
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    
    if uploaded_file is not None:
        file_processed_button = st.button("Process Statement")
        
        if file_processed_button:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            file_processed = True
            
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
                        # Initialize edited_transactions with new data
                        st.session_state.edited_transactions = [
                            {
                                "merchant": t.get("merchant", ""),
                                "amount": t.get("amount", 0.0),
                                "category": t.get("category", "others"),
                                "date": t.get("date", ""),
                                "include": True,
                                "type": t.get("type", "withdrawal")
                            } for t in transactions
                        ]
                        
                        # Show number of transactions processed
                        st.info(f"Successfully processed {len(transactions)} transactions from your statement.")
                        st.session_state.profile_saved = False
                except Exception as e:
                    st.error(f"Error processing statement: {str(e)}")
                    st.info("If you're having issues with statement upload, you can use manual entry instead.")
    
    # Display transaction editing interface if there are transactions to edit
    if 'edited_transactions' in st.session_state and len(st.session_state.edited_transactions) > 0:
        st.subheader("Review and Edit Transactions")
        st.info("You can review, edit, or remove transactions before finalizing your spending profile.")
        
        # Get all categories for dropdowns
        categories = merchant_categorizer.get_categories() if merchant_categorizer else [
            "groceries", "dining", "transportation", "shopping", 
            "entertainment", "travel", "utilities", "healthcare", 
            "education", "others"
        ]
        
        # Table headers
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
        with col1:
            st.markdown("**Merchant**")
        with col2:
            st.markdown("**Amount**")
        with col3:
            st.markdown("**Category**")
        with col4:
            st.markdown("**Date**")
        with col5:
            st.markdown("**Include**")
        
        # Create a container for the transactions (better for rerun performance)
        transaction_container = st.container()
        
        # Display each transaction in a row
        for i, transaction in enumerate(st.session_state.edited_transactions):
            with transaction_container:
                col1, col2, col3, col4, col5 = st.columns([4, 2, 3, 2, 1])
                
                with col1:
                    edited_merchant = st.text_input(
                        "", 
                        value=transaction["merchant"], 
                        key=f"merchant_{i}"
                    )
                    transaction["merchant"] = edited_merchant
                
                with col2:
                    edited_amount = st.number_input(
                        "", 
                        min_value=0.0, 
                        value=float(transaction["amount"]), 
                        step=0.01, 
                        key=f"amount_{i}"
                    )
                    transaction["amount"] = edited_amount
                
                with col3:
                    edited_category = st.selectbox(
                        "", 
                        categories, 
                        index=categories.index(transaction["category"]) if transaction["category"] in categories else 0,
                        key=f"category_{i}"
                    )
                    transaction["category"] = edited_category
                
                with col4:
                    # Replace st.text with a disabled text_input for consistent height
                    st.text_input("", value=transaction["date"], key=f"date_{i}", disabled=True)
                
                with col5:
                    # Adjust vertical position of checkbox with empty space to align it
                    st.write("")
                    include = st.checkbox("", value=transaction["include"], key=f"include_{i}")
                    
                    # Mark profile as needing update when include status changes
                    if transaction["include"] != include:
                        st.session_state.profile_saved = False
                    
                    transaction["include"] = include
        
        # Add New Transaction
        st.markdown("---")
        st.subheader("Add New Transaction")
        
        # Input fields for a new transaction
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.text_input("Merchant", key="new_merchant", value=st.session_state.new_merchant)
        
        with col2:
            st.number_input("Amount", min_value=0.0, value=st.session_state.new_amount, step=0.01, key="new_amount")
        
        with col3:
            st.selectbox("Category", categories, index=categories.index(st.session_state.new_category) if st.session_state.new_category in categories else 0, key="new_category")
        
        with col4:
            st.button("Add", on_click=add_transaction)
        
        # Add some space before the navigation/save buttons
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Submit button to generate spending profile - moved to bottom right
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("Save and Generate Spending Profile", type="primary"):
                generate_spending_profile()
                # Success message will show after rerun
        
        # Show success message if profile was just saved
        if 'show_success' in st.session_state and st.session_state.show_success:
            st.success("Spending profile saved! Please proceed to Recommendations.")
            # Reset success flag after showing
            st.session_state.show_success = False

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
            st.session_state.profile_saved = True
            st.success("Spending profile saved! Please proceed to Recommendations.")

# Display current spending profile if it exists
if 'spending_profile' in st.session_state and 'profile_saved' in st.session_state and st.session_state.profile_saved:
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Current Spending Profile")
    
    # Calculate total spending
    total_spending = sum(amount for amount in st.session_state.spending_profile.values())
    
    # Create spending data for the table
    spending_data = [
        {
            "Category": category.capitalize(),
            "Amount": f"${amount:.2f}",
            "Percentage": f"{(amount/total_spending*100):.1f}%" if total_spending > 0 else "0%"
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
        
        # Add a summary row
        st.markdown(f"**Total Spending:** ${total_spending:.2f}")
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