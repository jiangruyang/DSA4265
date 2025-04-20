import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Spending Input | Credit Card Optimizer",
    page_icon="💰",
    layout="wide"
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
# Import standardized components
from src.user_interface.components import page_header, section_header, subsection_header, progress_tracker, nav_buttons

# Ensure the application event loop is initialized
initialize_app_event_loop()

# Display progress tracker (Spending Input is step 1)
progress_tracker(current_step=1)

if "preferences" not in st.session_state or not st.session_state.preferences:
    st.warning("Please complete the User Profile first.")
    st.info("Go to the User Profile page to set your preferences.")
    # Add direct navigation button
    st.page_link("pages/1_User_Profile.py", label="Go to User Profile", icon="👤", use_container_width=False)
else:
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

    # User Interface - use standardized header
    page_header(
        title="Spending Input",
        icon="💰",
        description="Enter your spending information to help us recommend the best credit cards for your needs."
    )

    # Get merchant categorizer from session state
    merchant_categorizer = st.session_state.merchant_categorizer

    # Create a statement parser that uses the merchant categorizer
    statement_parser = PDFStatementParser()
    if merchant_categorizer:
        statement_parser.merchant_categorizer = merchant_categorizer

    section_header("Choose Your Input Method")
    input_method = st.radio(
        "How would you like to enter your spending information?",
        ["Upload Statements", "Manual Entry"],
        help="Upload your credit card statements or manually enter your spending by category"
    )

    # Initialize uploaded_file to None
    uploaded_file = None
    file_processed = False

    if input_method == "Upload Statements":
        subsection_header(
            title="Upload Credit Card Statements (PDF)",
            description="Upload your credit card statements in PDF format for automatic transaction categorization."
        )
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type="pdf", 
            help="Supported format: PDF statements from major banks"
        )
        
        if uploaded_file is not None:
            file_processed_button = st.button(
                "Process Statement", 
                type="primary", 
                use_container_width=True
            )
            
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
            subsection_header(
                title="Review and Edit Transactions",
                description="You can review, edit, or remove transactions before finalizing your spending profile."
            )
            
            # Get all categories for dropdowns
            categories = merchant_categorizer.get_categories() if merchant_categorizer else [
                "groceries", "dining", "transportation", "shopping", 
                "entertainment", "travel", "utilities", "healthcare", 
                "education", "others"
            ]
            
            # Use a container for the transaction table
            with st.container():
                # Table headers with a distinct background
                header_cols = st.columns([4, 2, 3, 2, 1])
                with header_cols[0]:
                    st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>Merchant</b></div>", unsafe_allow_html=True)
                with header_cols[1]:
                    st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>Amount</b></div>", unsafe_allow_html=True)
                with header_cols[2]:
                    st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>Category</b></div>", unsafe_allow_html=True)
                with header_cols[3]:
                    st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>Date</b></div>", unsafe_allow_html=True)
                with header_cols[4]:
                    st.markdown("<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;'><b>Include</b></div>", unsafe_allow_html=True)
                
                # Add a divider between headers and first row
                st.divider()
                
                # Create a container for the transactions (better for rerun performance)
                transaction_container = st.container()
                
                # Display each transaction in a row
                for i, transaction in enumerate(st.session_state.edited_transactions):
                    with transaction_container:
                        # Add a subtle divider between rows
                        if i > 0:
                            st.divider()
                            
                        col1, col2, col3, col4, col5 = st.columns([4, 2, 3, 2, 1])
                        
                        with col1:
                            edited_merchant = st.text_input(
                                f"Merchant {i+1}", 
                                value=transaction["merchant"], 
                                key=f"merchant_{i}",
                                label_visibility="collapsed"
                            )
                            transaction["merchant"] = edited_merchant
                        
                        with col2:
                            edited_amount = st.number_input(
                                f"Amount {i+1}", 
                                min_value=0.0, 
                                value=float(transaction["amount"]), 
                                step=0.01, 
                                key=f"amount_{i}",
                                label_visibility="collapsed"
                            )
                            transaction["amount"] = edited_amount
                        
                        with col3:
                            edited_category = st.selectbox(
                                f"Category {i+1}", 
                                categories, 
                                index=categories.index(transaction["category"]) if transaction["category"] in categories else 0,
                                key=f"category_{i}",
                                label_visibility="collapsed"
                            )
                            transaction["category"] = edited_category
                        
                        with col4:
                            # Replace st.text with a disabled text_input for consistent height
                            st.text_input(
                                f"Date {i+1}", 
                                value=transaction["date"], 
                                key=f"date_{i}", 
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        with col5:
                            # Adjust vertical position of checkbox with empty space to align it
                            st.write("")
                            include = st.checkbox(
                                f"Include transaction {i+1}", 
                                value=transaction["include"], 
                                key=f"include_{i}",
                                label_visibility="collapsed"
                            )
                            
                            # Mark profile as needing update when include status changes
                            if transaction["include"] != include:
                                st.session_state.profile_saved = False
                            
                            transaction["include"] = include
            
            # Add New Transaction section
            subsection_header("Add New Transaction")
            
            # Input fields for a new transaction
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.text_input("Merchant", key="new_merchant", value=st.session_state.new_merchant)
            
            with col2:
                st.number_input("Amount", min_value=0.0, value=st.session_state.new_amount, step=0.01, key="new_amount")
            
            with col3:
                st.selectbox("Category", categories, index=categories.index(st.session_state.new_category) if st.session_state.new_category in categories else 0, key="new_category")
            
            with col4:
                st.button("Add", on_click=add_transaction, use_container_width=True)
            
            # Submit button to generate spending profile - moved to bottom right with improved styling
            st.divider()  # Use divider instead of spacing

            st.info("Please ensure you have added all transactions you wish to include in your spending profile before generating.")
            
            if st.button("Save and Generate Spending Profile", type="primary", use_container_width=True):
                generate_spending_profile()
                # Success message will show after rerun
            
            # Show success message if profile was just saved
            if 'show_success' in st.session_state and st.session_state.show_success:
                st.success("Spending profile saved! Please proceed to Recommendations.")
                # Reset success flag after showing
                st.session_state.show_success = False

    else:  # Manual Entry
        subsection_header(
            title="Enter Monthly Spending by Category",
            description="Enter your average monthly spending for each category."
        )
        
        with st.form("manual_spending_form"):
            # Get categories from categorizer
            categories = merchant_categorizer.get_categories() if merchant_categorizer else [
                "groceries", "dining", "transportation", "shopping", 
                "entertainment", "travel", "utilities", "healthcare", 
                "education", "others"
            ]
            
            # Use columns for a more compact layout
            col1, col2 = st.columns(2)
            
            # Create input fields for each category, alternating between columns
            spending_inputs = {}
            for i, category in enumerate(categories):
                with col1 if i % 2 == 0 else col2:
                    spending_inputs[category] = st.number_input(
                        f"{category.capitalize()} (SGD)",
                        min_value=0.0,
                        value=st.session_state.spending_profile.get(category, 0.0),
                        step=10.0,
                        help=f"Your average monthly spending on {category}"
                    )
            
            # Use primary button type and container width
            submit_spending = st.form_submit_button(
                "Save Spending Profile", 
                type="primary", 
                use_container_width=True
            )
            
            if submit_spending:
                st.session_state.spending_profile = spending_inputs
                st.session_state.profile_saved = True
                st.success("Spending profile saved! Please proceed to Recommendations.")

    # Display current spending profile if it exists
    if 'spending_profile' in st.session_state and 'profile_saved' in st.session_state and st.session_state.profile_saved:
        section_header(
            title="Current Spending Profile",
            description="This is your current spending profile based on the data you've entered."
        )
        
        # Calculate total spending
        total_spending = sum(amount for amount in st.session_state.spending_profile.values())
        
        # Display a summary of total spending
        st.metric(
            label="Total Monthly Spending", 
            value=f"${total_spending:.2f}"
        )

        # Add a header
        st.subheader("Spending by Category")
        
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
            # Create a styled dataframe
            st.dataframe(
                pd.DataFrame(spending_data),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Your current spending profile has no spending in any category.")
        
        # Display a bar chart for spending distribution if there's data
        if total_spending > 0:
            # Prepare data for bar chart - sort by amount descending
            spending_data = {}
            for category, amount in sorted(st.session_state.spending_profile.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True):
                if amount > 0:  # Only include categories with spending
                    spending_data[category.capitalize()] = amount
            
            if spending_data:
                # Create a DataFrame for plotly chart
                chart_df = pd.DataFrame({
                    'Category': list(spending_data.keys()),
                    'Amount': list(spending_data.values())
                })
                
                # Sort by amount descending
                chart_df = chart_df.sort_values('Amount', ascending=False)
                
                # Create a plotly horizontal bar chart
                import plotly.express as px
                fig = px.bar(
                    chart_df,
                    x='Amount',
                    y='Category',
                    orientation='h',
                    labels={'Amount': 'Amount ($)', 'Category': ''},
                    text_auto='.2f'
                )
                
                # Customize layout
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                # Display the plotly chart
                st.plotly_chart(fig, use_container_width=True)
        
        st.info("Proceed to the Recommendations page to generate credit card recommendations based on your spending profile.")

# Navigation buttons with standardized component
st.divider()
nav_buttons(
    prev_page="pages/1_User_Profile.py", 
    next_page="pages/3_Recommendations.py", 
    next_condition=bool('spending_profile' in st.session_state and st.session_state.spending_profile),
    next_warning="Please save your spending profile before proceeding."
) 