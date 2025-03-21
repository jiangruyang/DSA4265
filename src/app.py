import os
import streamlit as st
from dotenv import load_dotenv
import io
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.categorization.statement_parser import StatementParser
from src.categorization.transaction_classifier import TransactionClassifier
from src.synergy_engine.synergy_calculator import SynergyCalculator
from src.synergy_engine.card_database import CardDatabase
from src.chat_agent.llm_config import get_llm, get_chat_prompt, format_chat_history
from src.rag_pipeline import DocumentLoader, RAGChat

# Import T&C modules if available
try:
    from src.tc_summaries import TCProvider
    TC_PROVIDER_AVAILABLE = True
    logger.info("T&C Summaries module available")
except ImportError:
    logger.warning("T&C Summaries module not available")
    TC_PROVIDER_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Credit Card Rewards Optimizer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Credit Card Rewards Optimizer: Singapore Edition")
st.markdown("""
This application helps you optimize your credit card usage by analyzing your spending patterns
and recommending the best card combinations for maximizing rewards.
""")

# Session state initialization
if "spending_data" not in st.session_state:
    st.session_state.spending_data = None
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# Initialize TC Provider if available
if "tc_provider" not in st.session_state and TC_PROVIDER_AVAILABLE:
    try:
        st.session_state.tc_provider = TCProvider()
        logger.info(f"Loaded {len(st.session_state.tc_provider.get_card_names())} card summaries")
    except Exception as e:
        logger.error(f"Error initializing TC Provider: {str(e)}")
        st.session_state.tc_provider = None

# Initialize RAG pipeline
if "rag_vector_store" not in st.session_state:
    try:
        document_loader = DocumentLoader(vector_db_path="./data/vector_db")
        tc_directory = "./data/card_tcs/pdf"
        if os.path.exists(tc_directory):
            st.session_state.rag_vector_store = document_loader.process_and_load(tc_directory)
            logger.info(f"Loaded T&C documents from {tc_directory}")
        else:
            logger.warning(f"T&C directory not found: {tc_directory}")
            st.session_state.rag_vector_store = None
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        st.session_state.rag_vector_store = None

if "rag_chat" not in st.session_state and st.session_state.rag_vector_store is not None:
    try:
        st.session_state.rag_chat = RAGChat(st.session_state.rag_vector_store, temperature=0.7)
        logger.info("RAG Chat initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG Chat: {str(e)}")
        st.session_state.rag_chat = None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["User Profile", "Spending Analysis", "Recommendations", "Chat with AI"]
)

# Page content
if page == "User Profile":
    st.header("User Profile")
    
    st.subheader("Reward Preferences")
    reward_preference = st.selectbox(
        "What type of rewards do you prefer?",
        ["Miles", "Cashback", "Points", "No Preference"]
    )
    
    annual_fee_tolerance = st.slider(
        "Annual Fee Tolerance (SGD)",
        0, 1000, 200, 50
    )
    
    income_range = st.selectbox(
        "Income Range (SGD)",
        ["Below 30,000", "30,000 - 50,000", "50,000 - 80,000", "80,000 - 120,000", "Above 120,000"]
    )
    
    if st.button("Save Preferences"):
        st.session_state.user_preferences = {
            "reward_preference": reward_preference,
            "annual_fee_tolerance": annual_fee_tolerance,
            "income_range": income_range
        }
        st.success("Preferences saved successfully!")

elif page == "Spending Analysis":
    st.header("Spending Analysis")
    
    input_method = st.radio(
        "How would you like to input your spending data?",
        ["Upload Statement", "Manual Entry"]
    )
    
    if input_method == "Upload Statement":
        uploaded_file = st.file_uploader("Upload your credit card statement (PDF)", type=["pdf"])
        if uploaded_file:
            with st.spinner("Processing your statement..."):
                try:
                    # Read file content
                    file_content = uploaded_file.read()
                    file_format = "pdf"
                    
                    # Parse statement
                    parser = StatementParser()
                    transactions = parser.parse_statement(file_content, file_format)
                    
                    # Log the parsed transactions
                    logger.debug(f"Parsed {len(transactions)} transactions from PDF")
                    
                    if not transactions:
                        st.error("No transactions could be extracted from the PDF. Please try a different file or use manual entry.")
                    else:
                        # Classify transactions
                        classifier = TransactionClassifier()
                        classified_transactions = classifier.classify_transactions(transactions)
                        
                        # Aggregate by category
                        category_totals = classifier.aggregate_by_category(classified_transactions)
                        
                        # Display transaction table
                        st.subheader("Extracted Transactions")
                        
                        # Convert to DataFrame for display - handle case where data doesn't have all expected columns
                        import pandas as pd
                        
                        # Create a DataFrame with the correct columns
                        df_data = []
                        for transaction in classified_transactions:
                            df_data.append({
                                'date': transaction.get('date', ''),
                                'description': transaction.get('description', ''),
                                'amount': transaction.get('amount', 0.0),
                                'category': transaction.get('category', 'Others')
                            })
                        
                        df = pd.DataFrame(df_data)
                        
                        if not df.empty:
                            st.dataframe(df)
                        else:
                            st.warning("No transaction data to display")
                        
                        # Display category totals
                        st.subheader("Spending by Category")
                        
                        # Create two columns for the chart and the data
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # Convert category totals to a format for chart
                            chart_data = pd.DataFrame({
                                'Category': list(category_totals.keys()),
                                'Amount': list(category_totals.values())
                            })
                            chart_data = chart_data[chart_data['Amount'] > 0]  # Only show categories with spending
                            
                            # Display chart
                            st.bar_chart(chart_data.set_index('Category'))
                        
                        with col2:
                            # Display category totals in a table
                            for category, amount in category_totals.items():
                                if amount > 0:
                                    st.write(f"**{category}:** ${amount:.2f}")
                        
                        # Store the spending data in session state
                        st.session_state.spending_data = {
                            **{k.lower(): v for k, v in category_totals.items()},
                            "total": sum(category_totals.values())
                        }
                        
                        # Provide button to proceed to recommendations
                        if st.button("Proceed to Recommendations"):
                            st.session_state.update({"_radio": "Recommendations"})
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error processing statement: {str(e)}")
                    logger.exception("Error processing PDF statement")
                    st.info("Please try again or use manual entry.")
    
    else:  # Manual Entry
        st.subheader("Enter your monthly spending by category")
        
        dining = st.number_input("Dining (SGD)", min_value=0, value=0)
        groceries = st.number_input("Groceries (SGD)", min_value=0, value=0)
        transport = st.number_input("Transport (SGD)", min_value=0, value=0)
        shopping = st.number_input("Shopping (SGD)", min_value=0, value=0)
        travel = st.number_input("Travel (SGD)", min_value=0, value=0)
        bills = st.number_input("Bills & Utilities (SGD)", min_value=0, value=0)
        others = st.number_input("Others (SGD)", min_value=0, value=0)
        
        if st.button("Save Spending Data"):
            st.session_state.spending_data = {
                "dining": dining,
                "groceries": groceries,
                "transport": transport,
                "shopping": shopping,
                "travel": travel,
                "bills": bills,
                "others": others,
                "total": dining + groceries + transport + shopping + travel + bills + others
            }
            st.success("Spending data saved successfully!")
            st.info("You can now proceed to the Recommendations page.")

elif page == "Recommendations":
    st.header("Card Recommendations")
    
    if st.session_state.spending_data is None:
        st.warning("Please input your spending data first.")
        st.button("Go to Spending Analysis", on_click=lambda: st.session_state.update({"_radio": "Spending Analysis"}))
    
    elif st.session_state.user_preferences is None:
        st.warning("Please set your preferences first.")
        st.button("Go to User Profile", on_click=lambda: st.session_state.update({"_radio": "User Profile"}))
    
    else:
        with st.spinner("Generating recommendations based on your data..."):
            try:
                # Initialize synergy calculator
                card_db = CardDatabase()
                card_db.load_cards()
                calculator = SynergyCalculator(card_db)
                
                # Map user spending data to format expected by synergy calculator
                spending = {
                    key: float(value) for key, value in st.session_state.spending_data.items()
                    if key != "total"  # Skip the total field
                }
                
                # Map user preferences to format expected by synergy calculator
                user_preferences = {
                    "reward_preference": st.session_state.user_preferences["reward_preference"].lower(),
                    "annual_fee_tolerance": float(st.session_state.user_preferences["annual_fee_tolerance"]),
                    "income_range": st.session_state.user_preferences["income_range"]
                }
                
                # Find optimal card combination
                results = calculator.find_optimal_card_combination(
                    spending=spending,
                    user_preferences=user_preferences,
                    max_cards=2  # Limit to 2 cards for simplicity
                )
                
                # Log the results for debugging
                logger.debug(f"Synergy results: {results}")
                
                # Store recommendations in session state
                if results is None:
                    st.error("Unable to generate recommendations. No suitable cards were found based on your preferences and spending patterns.")
                    st.info("Try adjusting your preferences or adding more spending data.")
                    st.session_state.recommendations = []
                else:
                    st.session_state.recommendations = results["cards"]
                    
                    # Display results
                    st.subheader("Recommended Card Combination")
                    
                    # If we have multiple cards, show them in columns
                    cards = results["cards"]
                    cols = st.columns(len(cards))
                    
                    for i, card_name in enumerate(cards):
                        # Get the full card data from the database
                        card = None
                        for c in card_db.get_all_cards():
                            if c["name"] == card_name:
                                card = c
                                break
                        
                        if card:
                            with cols[i]:
                                st.markdown(f"#### {card['name']}")
                                st.markdown(f"**Issuer:** {card['issuer']}")
                                st.markdown(f"**Annual Fee:** ${card['annual_fee']['amount']:.2f}")
                                
                                # Show waiver condition if available
                                if "waiver_condition" in card["annual_fee"]:
                                    st.markdown(f"**Fee Waiver:** {card['annual_fee']['waiver_condition']}")
                                
                                st.markdown("**Reward Rates:**")
                                for category, rate in card["reward_rates"].items():
                                    if rate > 0:
                                        unit = "%" if card["reward_type"].lower() == "cashback" else "miles/$"
                                        st.markdown(f"- {category.capitalize()}: {rate} {unit}")
                
                # Display the usage strategy
                st.subheader("Usage Strategy")
                st.markdown(results["usage_strategy"])
                
                # Display reward summary
                st.subheader("Reward Summary")
                # Get reward unit with fallback
                reward_unit = results.get("reward_unit", "points")
                
                # Determine the reward type from the first card
                if not reward_unit and len(cards) > 0:
                    first_card = None
                    for c in card_db.get_all_cards():
                        if c["name"] == cards[0]:
                            first_card = c
                            break
                    
                    if first_card:
                        reward_unit = first_card.get("reward_type", "points").lower()
                        if "cashback" in reward_unit:
                            reward_unit = "cashback"
                        elif "mile" in reward_unit:
                            reward_unit = "miles"
                        else:
                            reward_unit = "points"
                
                # Display the total rewards
                total_reward = results.get("total_reward", 0.0)
                st.markdown(f"**Total Monthly Rewards:** {total_reward:.2f} {reward_unit}")
                
                if reward_unit == "cashback":
                    annual_savings = total_reward * 12
                    st.markdown(f"**Estimated Annual Savings:** ${annual_savings:.2f}")
                
                # Display spend breakdown
                spend_data = []
                for category, amount in spending.items():
                    if amount > 0:
                        spend_data.append({"Category": category.capitalize(), "Amount": amount})
                
                if spend_data:
                    import pandas as pd
                    spend_df = pd.DataFrame(spend_data)
                    st.subheader("Your Spending Breakdown")
                    st.bar_chart(spend_df.set_index("Category"))
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.info("Please check your spending data and preferences, then try again.")

elif page == "Chat with AI":
    st.header("Chat with AI Assistant")
    
    st.info("Ask questions about card T&Cs, scenario planning, or get clarification on recommendations.")
    
    # Check if RAG pipeline is initialized
    if st.session_state.rag_vector_store is None:
        st.warning("T&C document database is not initialized. Some answers about card terms may be less accurate.")
    else:
        st.success("T&C document database is loaded. Ask specific questions about card terms and conditions!")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source['card_name']} ({source['source']})")
    
    # Function to handle question submission
    def submit_question():
        if st.session_state.user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_question})
            
            # Process the question (this will happen after the rerun)
            st.session_state.process_question = True
            
            # Clear the input after submission
            st.session_state.user_question = ""
    
    # Function to set a suggested question
    def use_suggested_question(question):
        st.session_state.user_question = question
        submit_question()
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=False):
        st.text_input("Type your question here:", key="user_question")
        submit_button = st.form_submit_button("Send", on_click=submit_question)
    
    # Suggested questions
    st.markdown("### Suggested Questions")
    
    # Dynamic suggested questions based on recommendations and document context
    if st.session_state.recommendations and hasattr(st.session_state, 'rag_chat') and st.session_state.rag_chat:
        try:
            # Get user preferences and spending data
            user_prefs = st.session_state.user_preferences or {"reward_preference": "Miles", "annual_fee_tolerance": 200}
            spend_data = st.session_state.spending_data or {"dining": 500, "shopping": 300}
            
            # Generate dynamic questions using RAG
            questions = st.session_state.rag_chat.generate_suggested_questions(
                user_preferences=user_prefs,
                spending_data=spend_data,
                recommended_cards=st.session_state.recommendations
            )
        except Exception as e:
            logger.error(f"Error generating suggested questions: {str(e)}")
            questions = [
                "What are the main benefits of the POSB Everyday Card?",
                "How do I earn rebates with the POSB Everyday Card?",
                "Is there a cap on rebates for the POSB Everyday Card?",
                "What are the annual fee waiver conditions for this card?"
            ]
    else:
        questions = [
            "What are the main benefits of the POSB Everyday Card?",
            "How do I earn rebates with the POSB Everyday Card?",
            "Is there a cap on rebates for the POSB Everyday Card?",
            "What are the annual fee waiver conditions for this card?"
        ]
    
    cols = st.columns(2)
    for i, question in enumerate(questions):
        cols[i % 2].button(question, key=f"q_{i}", on_click=use_suggested_question, args=(question,))
    
    # Process the user's question if it was just submitted
    if st.session_state.get("process_question", False):
        # Reset the flag
        st.session_state.process_question = False
        
        # Get the last user question from chat history
        user_input = st.session_state.chat_history[-1]["content"]
        
        # Get AI response
        with st.spinner("Thinking..."):
            try:
                # Check if we should use RAG or regular LLM
                if hasattr(st.session_state, 'rag_chat') and st.session_state.rag_chat and ("card" in user_input.lower() or "t&c" in user_input.lower() or "term" in user_input.lower() or "condition" in user_input.lower() or "rebate" in user_input.lower() or "miles" in user_input.lower() or "cashback" in user_input.lower()):
                    # Use RAG for T&C specific questions
                    logger.info("Using RAG pipeline for user query")
                    response = st.session_state.rag_chat.generate_response(user_input)
                    ai_response = response["response"]
                    
                    # Add sources metadata to the chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": ai_response,
                        "sources": response["sources"]
                    })
                else:
                    # Use regular LLM for general questions
                    logger.info("Using regular LLM for user query")
                    messages = [{"role": "system", "content": get_chat_prompt()}]
                    
                    # Add previous chat history (excluding the most recent user question)
                    chat_history_for_context = st.session_state.chat_history[:-1] if st.session_state.chat_history else []
                    messages.extend(chat_history_for_context)
                    
                    # Get recommendation context if available
                    if st.session_state.recommendations:
                        # Initialize card database to get full card details
                        card_db = CardDatabase()
                        card_db.load_cards()
                        
                        card_context = "Based on your profile, we recommended: "
                        
                        # The recommendations are strings (card names), not dictionaries
                        for card_name in st.session_state.recommendations:
                            # Find the full card data
                            card = None
                            for c in card_db.get_all_cards():
                                if c["name"] == card_name:
                                    card = c
                                    break
                            
                            if card:
                                card_context += f"\n- {card['name']} ({card['issuer']})"
                            else:
                                card_context += f"\n- {card_name}"
                        
                        messages.append({"role": "system", "content": card_context})
                    
                    # Add user query
                    formatted_messages = format_chat_history(messages)
                    formatted_messages.append({"role": "user", "content": user_input})
                    
                    # Get response from LLM
                    llm = get_llm(temperature=0.7)
                    response = llm.invoke(formatted_messages)
                    
                    ai_response = response.content
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                ai_response = f"I'm sorry, I encountered an error: {str(e)}. Please make sure your OpenAI API key is set in the .env file."
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Force page to update to display new message
            st.rerun()

# Footer
st.markdown("---")
st.markdown("© 2023 Credit Card Rewards Optimizer | Powered by Generative AI") 