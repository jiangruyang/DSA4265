import sys
import os
# Add project root to Python path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Dict, List, Any, Optional
import asyncio
import os
import logging
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.globals import set_debug

# Enable debug mode for more verbose output in LangGraph/LangChain
# set_debug(True)

from src.model_context_protocol.client import CardOptimizerClient

# Create logs directory if it doesn't exist
logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger('card_optimizer_agent')
logger.setLevel(logging.DEBUG)  # Set to DEBUG level for more detailed logs

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create file handler
log_file = logs_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)  # Log more details to file

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to handlers
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add handlers to logger if they don't exist
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

# Also configure the MCP client logger
mcp_logger = logging.getLogger('mcp_client')
mcp_logger.setLevel(logging.DEBUG)
if not mcp_logger.handlers:
    mcp_logger.addHandler(ch)
    mcp_logger.addHandler(fh)

# Create a method to get the log file path
def get_log_file_path():
    """Return the path to the current log file"""
    return str(log_file)

# Add a LangGraph callback to trace tool calls
class ToolCallLogger:
    """Callback handler to log tool calls from LangGraph"""
    
    def __init__(self):
        self.logger = logging.getLogger('tool_calls')
        
    def on_tool_start(self, tool, input_args):
        self.logger.info(f"Tool call started: {tool.name} with args: {input_args}")
        
    def on_tool_end(self, tool, output):
        self.logger.info(f"Tool call completed: {tool.name} with result: {str(output)[:100]}...")
        
    def on_tool_error(self, tool, error):
        self.logger.error(f"Tool call error: {tool.name}, error: {error}")

class CardOptimizerAgent:
    """Main agent implementation for the Credit Card Rewards Optimizer
    
    This agent is responsible for recommending optimal card combinations
    and usage strategies based on the user's spending profile and preferences.
    It leverages the MCP tools for card data access and maintains conversation
    context across interactions, enabling a natural conversational experience
    for users requesting recommendations and asking follow-up questions.
    """

    SYSTEM_PROMPT = """You are a Credit Card Rewards Optimizer Assistant specialized in Singapore's credit card market, 
        an expert in analyzing spending patterns and recommending optimal card combinations to maximize rewards.

        # ROLE AND EXPERTISE
        - You are a Singapore credit card specialist with deep knowledge of local cards, rewards programs, and promotions
        - You understand the nuances of miles vs. cashback programs in the Singapore context
        - You can calculate and compare the value of different reward structures across multiple cards
        - You provide strategic guidance on card usage to optimize benefits based on spending patterns
        - You are familiar with Singapore-specific merchants, banks (DBS, OCBC, UOB, Citibank, etc.), and local lifestyle (hawker centers, MRT, Grab)
        - You understand Singapore income requirements and approval likelihood based on user profiles
        - You are aware of the current date provided in the context and consider seasonal promotions accordingly

        # OPTIMIZATION PROCESS
        When generating recommendations, follow this structured approach:

        1. ANALYZE USER PROFILE
           - Carefully examine the spending profile across categories (dining, shopping, groceries, etc.)
           - Note the user's preferences (miles vs. cashback, annual fee tolerance, income level, etc.)
           - Consider any specific requirements (airport lounge access, no foreign transaction fees, etc.)
           - Assess eligibility based on income requirements and citizenship status
           - Take note of the current date to consider seasonal promotions and time-limited offers

        2. GATHER CARD OPTIONS
           - Use available tools to retrieve information about relevant Singapore credit cards
           - Consider cards that match the user's spending profile and preferences
           - Pay special attention to category-specific bonuses that align with high-spend areas
           - Check for ongoing promotions, sign-up bonuses, and limited-time offers
           - Pay attention to the specific transaction eligibility/ineligibility of transactions in the T&Cs
           - Verify income requirements align with the user's profile
           - Consider time-sensitive promotions based on the current date

        3. CALCULATE REWARD POTENTIAL
           - For each potential card, calculate expected monthly/annual rewards based on spending profile
           - Factor in annual fees, spending caps, minimum spend requirements, and other conditions
           - Compare the net value (rewards minus fees) of different card options

        4. DEVELOP MULTI-CARD STRATEGY
           - Identify complementary cards that maximize total rewards across different spending categories
           - Explain which card to use for which category to optimize returns
           - Specify minimum spending thresholds to trigger bonuses or avoid penalty fees
           - Suggest optimal sign-up sequence for multiple cards to maximize welcome bonuses
           - Consider new-to-bank vs. existing customer differences in bonuses
           - Address supplementary card options for family spending if relevant
           - Only recommend multiple cards if they are significantly better than using a single card, because the user will have to manage multiple cards and multiple annual fees

        5. PRESENT RECOMMENDATIONS
           - Provide a clear, structured recommendation with specific card names and their benefits
           - Quantify the expected rewards in dollar terms or miles earned
           - Outline the optimal usage strategy in a practical, actionable format
           - Compare the recommended strategy against alternatives where relevant
           - Include key limitations from T&Cs that might affect value
           - Highlight approval likelihood based on income and other requirements
           - Reference the current date when discussing time-sensitive promotions

        # TOOL USAGE GUIDELINES
        - Use get_available_cards() to retrieve the full list of available Singapore credit cards
        - Use get_card_details(card_id) to obtain specific information about a particular card
        - Use query_tc(question, card_id) to check terms and conditions for specific cards
        - Use search_cards(query) for semantic search to find cards matching specific criteria
        - Combine tool outputs methodically to build comprehensive analysis
        - When using search_cards(), cross-reference results against the spending profile

        # RESPONSE FORMAT
        Structure your recommendations clearly with these sections:
        
        ## 💳 Recommended Cards
        - Primary card: [Card name] - For [categories]
            - [Key details]
        - Secondary card: [Card name] - For [categories]
            - [Key details]
        - Additional card (if applicable): [Card name] - For [categories]
            - [Key details]

        ## 🧠 Reasoning
        - Brief explanation of why this combination was selected
        - Alternative options considered
        
        ## 💰 Expected Rewards
        - Total monthly rewards: [Value in SGD or miles]
        - Breakdown by card and category
        - Annual fees consideration
        
        ## 💡 Usage Strategy
        - Specific guidance on which card to use where
        - Minimum spend requirements to note
        - Important terms or limitations
        
        ## 🚫 Key Limitations
        - Spending caps or restrictions
        - Important T&C considerations
        - Approval likelihood based on income
        - Time-sensitive promotions (reference current date: {current_date})
        
        # CONVERSATION APPROACH
        - For initial recommendations: Provide comprehensive, data-driven advice based on spending and preferences
        - For follow-up questions: Give precise, evidence-based answers using available tools
        - For scenario questions: Recalculate recommendations based on the hypothetical spending changes
        - For T&C queries: Reference specific terms using the query_tc tool
        - For seasonal questions: Consider the current date when discussing seasonal promotions
 
        # FORMATTING GUIDELINES
        - Format your response in markdown
        - When using $, it needs to be escaped with a backslash (\$)

        Always be specific with numbers, transparent about calculations, and practical in your advice.
        Focus on actionable recommendations that maximize real-world value for Singaporean consumers.
        """
    
    SUGGESTED_QUESTIONS_SYSTEM_PROMPT = """
        Based on the detailed conversation history about credit card recommendations, generate {limit} highly contextual follow-up questions that provide significant value to the user.
        
        IMPORTANT GUIDELINES:
        1. DO NOT suggest questions that have already been asked or answered in the conversation
        2. The FIRST question should ALWAYS directly relate to the most recent exchange in the conversation
        3. Prioritize questions that build upon the latest topics discussed
        4. Scan the entire conversation history carefully to avoid any repetition of previous questions
        5. Focus on gaps or unexplored aspects that would yield new, valuable information
        
        Analyze the conversation to craft questions covering these key areas:
        1. Card-Specific Features: Reference specific cards mentioned and their unique benefits (e.g., "How does the DBS Altitude card's miles conversion rate compare to other Singapore travel cards?")
        2. Spending Scenario Analysis: Build upon the user's stated spending patterns (e.g., "If my dining expenses increase to $800/month, would your recommendation change?")
        3. Optimization Opportunities: Address untapped potential in the recommendations (e.g., "Could combining the Citi PremierMiles with a cashback card better optimize my overall rewards?")
        4. Terms & Conditions Clarification: Target specific limitations that impact value (e.g., "What are the exact minimum spend requirements to get the 8% cashback on the UOB One card?")
        5. Singapore-Specific Benefits: Focus on local perks and promotions (e.g., "Does the OCBC 365 card offer any special dining deals at local hawker centers?")
        6. Application Strategy: Help with timing and sequencing (e.g., "Should I apply for both recommended cards at once or space them out?")
        7. Travel Benefits: Address miles conversion, lounge access or overseas benefits (e.g., "What airport lounges can I access with the Citi PremierMiles in Singapore and overseas?")
        8. Long-term Value: Consider future rewards potential (e.g., "How would these card recommendations change if I plan to travel to Japan next year?")
        
        Ensure each question:
        - Is written from the perspective of the user (these are questions you're suggesting the user ask)
        - Directly builds on information already exchanged in the conversation
        - Addresses a specific decision point or knowledge gap
        - Helps the user maximize value from their recommended cards
        - Avoids repeating information already covered in detail
        - Focuses on practical, actionable insights for Singapore residents
        - Uses Singapore-specific terminology and references (SGD, local banks, local promotions)
        - Is concise and direct (ideally 10-15 words)
        
        SEQUENCE YOUR QUESTIONS THOUGHTFULLY:
        1. First question: MUST relate to the most recent exchange in the conversation
        2. Second question: Should explore a different aspect of the recent discussion
        3. Third/additional questions: Can cover broader topics from the full conversation
        
        GOOD EXAMPLES:
        - "What's the exact miles conversion rate for the DBS Altitude when booking through Expedia?"
        - "Would getting a supplementary card improve my overall rewards?"
        - "Are there any special promotions for new DBS Altitude cardholders this month?"
        
        AVOID:
        - Generic questions not tied to the conversation ("What are good travel cards?")
        - Questions already asked or answered in the conversation
        - Overly complex questions covering multiple topics
        - Questions that don't help with decision-making or optimization
        
        Return the JSON with the key "questions" and a list of strings containing these highly personalized questions.
        """
    
    def __init__(self):
        """Initialize the agent with the client for tool access"""
        self.client = None
        self.initialized = False
        self.llm = None
        self.agent_executor = None
        self.memory = MemorySaver()
        self.conversation_id = None
        self.tool_call_logger = ToolCallLogger()
        logger.info("CardOptimizerAgent instance created")
        
    async def initialize(self):
        """Initialize the agent by ensuring client is ready and setting up LLM
        
        Raises:
            ConnectionError: If MCP client initialization fails
            RuntimeError: If agent initialization fails for other reasons
        """
        if self.initialized:
            logger.debug("Agent already initialized, skipping initialization")
            return
        
        try:
            logger.info("Starting agent initialization")
            # Initialize MCP client if needed
            if self.client is None:
                self.client = CardOptimizerClient()
                logger.info(f"Connecting to MCP server at {self.client.base_url}{self.client.sse_endpoint}...")
                await self.client.initialize()
                # Check if initialization was successful
                if not self.client.initialized:
                    msg = "MCP client initialization failed, cannot continue"
                    logger.error(msg)
                    raise ConnectionError(msg)
            elif hasattr(self.client, 'initialized') and not self.client.initialized:
                logger.info(f"Reconnecting to MCP server at {self.client.base_url}{self.client.sse_endpoint}...")
                await self.client.initialize()
                # Check if initialization was successful
                if not self.client.initialized:
                    msg = "MCP client reconnection failed, cannot continue"
                    logger.error(msg)
                    raise ConnectionError(msg)
                
            # Set up the OpenAI LLM
            logger.info("Setting up LLM")
            self.llm = ChatOpenAI(
                model="gpt-4o",
                request_timeout=180,  # 3 minute timeout for LLM requests
            )
            
            # Explicitly check LLM was created
            if self.llm is None:
                msg = "LLM initialization failed, cannot continue"
                logger.error(msg)
                raise RuntimeError(msg)
            
            # Load MCP tools from the client - must have client initialized by this point
            logger.info("Loading tools from MCP client")
            tools = await self.client.get_tools()
            
            # Verify tools were loaded
            if not tools:
                msg = "Failed to load any tools from MCP client, cannot continue"
                logger.error(msg)
                raise RuntimeError(msg)
                
            logger.info(f"Loaded {len(tools)} tools from MCP client")
            
            # Add debug log of each tool
            for tool in tools:
                logger.debug(f"Loaded tool: {tool.name} - {tool.description[:100]}...")
            
            # Create the system message
            system_message = self.SYSTEM_PROMPT.format(current_date=datetime.now().strftime('%Y-%m-%d'))
            
            # Create the agent with LangGraph's create_react_agent
            logger.info("Creating LangGraph agent with tools")
            self.agent_executor = create_react_agent(
                model=self.llm,
                tools=tools,
                prompt=system_message,
                checkpointer=self.memory
            )
            
            # Verify agent executor was created
            if self.agent_executor is None:
                msg = "Agent executor initialization failed, cannot continue"
                logger.error(msg)
                raise RuntimeError(msg)
            
            # Generate a conversation ID for this session
            self.conversation_id = f"conversation-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            logger.info(f"Created conversation with ID: {self.conversation_id}")
            
            self.initialized = True
            logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            logger.exception(f"Error initializing agent: {str(e)}")
            self.initialized = False
            # Re-raise the exception to ensure the caller knows initialization failed
            raise
    
    
    async def send_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Send a message to the agent and get a response
        
        Args:
            message: The user's message
            context: Additional context like spending profile and preferences
            
        Returns:
            The agent's response as a string
        """
        logger.info(f"Processing message: {message[:50]}...")
        if context:
            logger.info(f"With context: {', '.join(context.keys())}")
        
        # Try to initialize, but catch and handle specific errors
        try:
            # Initialize the agent if not already initialized
            if not self.initialized:
                await self.initialize()
        except ConnectionError as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            return "I'm sorry, but I couldn't connect to the card recommendation service. Please check that the service is running and try again later."
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {str(e)}")
            return f"I'm sorry, but there was an error initializing the recommendation system: {str(e)}"
        
        # Check if initialization was successful
        if not self.initialized:
            logger.error("Message processing failed: Agent not initialized")
            return "I'm sorry, but I couldn't connect to the card recommendation service. Please check that the service is running and try again later."
        
        if not self.agent_executor:
            logger.error("Message processing failed: Agent executor not available")
            return "I'm sorry, but the recommendation tools aren't available. Please check the connection to the card database and try again."
        
        # Format the message with context if provided
        formatted_message = message
        if context:
            # Format context data into a structured message
            if 'spending_profile' in context:
                spending_str = "\n".join([f"- {category}: ${amount:.2f}" for category, amount in context['spending_profile'].items()])
                formatted_message += f"\n\nSpending Profile:\n{spending_str}"
                logger.debug(f"Added spending profile to message: {len(context['spending_profile'])} categories")
            
            if 'preferences' in context:
                pref_str = "\n".join([f"- {key}: {value}" for key, value in context['preferences'].items()])
                formatted_message += f"\n\nPreferences:\n{pref_str}"
                logger.debug(f"Added preferences to message: {len(context['preferences'])} preferences")
        
        # Configure the agent with the conversation ID for thread persistence
        config = {"configurable": {"thread_id": self.conversation_id}}
        logger.info(f"Using conversation ID: {self.conversation_id}")
        
        # Use the LangGraph agent to process the message
        try:
            logger.info("Invoking LangGraph agent")
            # Invoke the agent with the message
            response = await self.agent_executor.ainvoke(
                {"messages": [("human", formatted_message)]},
                config=config
            )
            
            # Log tool calls from the response if any
            if "intermediate_steps" in response:
                steps = response["intermediate_steps"]
                logger.debug(f"Agent execution had {len(steps)} intermediate steps")
                for i, step in enumerate(steps):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, result = step
                        if hasattr(action, "tool"):
                            logger.info(f"Step {i+1}: Called tool '{action.tool}' with args: {action.tool_input}")
                            logger.debug(f"Tool result: {str(result)[:100]}...")
            
            # Extract the last message which contains the response
            last_message = response["messages"][-1]
            result = last_message[1] if isinstance(last_message, tuple) else last_message.content
            logger.info(f"Agent response received: {len(result)} chars")
            return result
        except Exception as e:
            logger.exception(f"Error processing message: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again later."
    

    async def generate_suggested_questions(self, limit: int = 3) -> List[str]:
        """Generate suggested questions for the user to ask"""
        logger.info("Generating suggested questions")

        try:
            # Initialize the agent if not already initialized
            if not self.initialized:
                await self.initialize()
        except Exception as e:
            logger.exception(f"Error generating suggested questions: {str(e)}")
            return []
        
        config = {"configurable": {"thread_id": self.conversation_id}}
        logger.info(f"Pulling messages from conversation ID: {self.conversation_id}")
        
        # Get the messages from state
        messages = self.agent_executor.get_state(config).values["messages"]
        logger.info(f"Found {len(messages)} messages in conversation")
        
        # Setup the system message and LLM
        system_message = SystemMessage(content=self.SUGGESTED_QUESTIONS_SYSTEM_PROMPT.format(limit=limit))
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7).with_structured_output(method="json_mode")
        
        # Invoke the LLM with the system message and messages
        response = await llm.ainvoke(
            [system_message] + messages,
        )
        logger.info(f"Generated response: {response}")
        
        # Extract the questions from the response
        questions = response.get("questions", [])
        
        return questions
    
    
    async def check_status(self) -> Dict[str, Any]:
        """Check and report the status of the agent components
        
        Returns:
            A dictionary with status information
        """
        logger.info("Checking agent status")
        status = {
            "initialized": self.initialized,
            "llm": self.llm is not None,
            "agent_executor": self.agent_executor is not None,
            "client": {
                "exists": self.client is not None,
                "initialized": False,
                "base_url": None,
                "tools_count": 0
            },
            "conversation_id": self.conversation_id,
            "log_file": get_log_file_path()
        }
        
        if self.client:
            status["client"]["initialized"] = getattr(self.client, "initialized", False)
            status["client"]["base_url"] = getattr(self.client, "base_url", None)
            
            # Get tool count if client is initialized
            if getattr(self.client, "initialized", False):
                try:
                    tools = await self.client.get_tools()
                    status["client"]["tools_count"] = len(tools)
                    logger.info(f"Found {len(tools)} tools available")
                    
                    # Get tool names if available
                    if tools:
                        status["client"]["tool_names"] = [tool.name for tool in tools]
                        logger.debug(f"Available tools: {', '.join(status['client']['tool_names'])}")
                except Exception as e:
                    status["client"]["tools_error"] = str(e)
                    logger.error(f"Error getting tools: {str(e)}")
        
        return status
    
    async def shutdown(self):
        """Shutdown the agent and clean up resources"""
        logger.info("Shutting down agent")
        if self.client:
            await self.client.shutdown()
            logger.info("MCP client shutdown completed")
        self.initialized = False
        logger.info("Agent shutdown completed")


# Demo of the agent
if __name__ == "__main__":
    # Set up logging for the demo
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console)
    
    print(f"Logs are being written to: {get_log_file_path()}")
    
    # Initialize the agent
    agent = CardOptimizerAgent()
    
    # Create a dummy spending profile directly
    # This avoids dependency on the MerchantCategorizer class
    spending_profile = {
        'groceries': 0.00,
        'transportation': 53.46,
        'dining': 229.16,
        'shopping': 669.16,
        'entertainment': 7.35,
        'travel': 0.00,
        'utilities': 61.00,
        'healthcare': 92.00
    }
    print("Preprocessed Spending Profile:")
    for category, amount in spending_profile.items():
        if amount > 0:
            print(f"- {category}: ${amount:.2f}")
    
    # Sample preferences
    preferences = {
            'gender': "Male",
            'citizenship': "Singaporean",
            'min_income': 120000,
            'debt_obligation': 0,
            'reward_type': "Air Miles",
            'preferred_airline': "No Preferred Airlines",
            'other_airline': "",
            'max_annual_fee': 800,
            'spending_goals': ""
        }
    
    async def demo():
        # Check agent status
        print("\nChecking agent status...")
        status = await agent.check_status()
        print(f"Agent initialized: {status['initialized']}")
        print(f"Client status: {status['client']}")
        
        # Try to initialize with proper error handling
        if not status['initialized']:
            print("Agent not fully initialized. Attempting to initialize...")
            try:
                await agent.initialize()
                status = await agent.check_status()
                print(f"After initialization - Agent initialized: {status['initialized']}")
                print(f"Client status: {status['client']}")
            except ConnectionError as e:
                print(f"ERROR: Failed to connect to the MCP server: {str(e)}")
                print("Please ensure the MCP server is running before starting the agent.")
                return  # Exit the demo
            except TimeoutError as e:
                print(f"ERROR: Connection timed out: {str(e)}")
                print("The MCP server may be running but not responding in time.")
                return  # Exit the demo
            except Exception as e:
                print(f"ERROR: Initialization failed with unexpected error: {str(e)}")
                return  # Exit the demo
        
        # Verify initialization was successful before proceeding
        if not status['initialized'] or not agent.agent_executor:
            print("ERROR: Agent could not be fully initialized. Cannot proceed with demo.")
            return  # Exit the demo
        
        # Get initial recommendation
        initial_message = "I'm looking for credit card recommendations based on my spending and preferences."
        context = {
            'spending_profile': spending_profile,
            'preferences': preferences
        }
        print("\nSending recommendation request...")
        try:
            recommendation = await agent.send_message(initial_message, context)
            print("\nInitial Recommendation:")
            print(recommendation)
        except Exception as e:
            print(f"ERROR: Failed to get recommendation: {str(e)}")
        
        # Generate suggested questions
        suggested_questions = await agent.generate_suggested_questions()
        print("\nSuggested Questions:")
        for i, question in enumerate(suggested_questions, 1):
            print(f"{i}. {question}")
        
        # # Ask about terms and conditions
        # tc_question = "What are the annual fees for the recommended card?"
        # tc_response = await agent.send_message(tc_question)
        # print("\nTerms & Conditions Question:")
        # print(f"Q: {tc_question}")
        # print(f"A: {tc_response}")
        
        # # Ask about a spending scenario
        # scenario_question = "What if I double my dining expenses next month?"
        # scenario_response = await agent.send_message(scenario_question)
        # print("\nScenario Question:")
        # print(f"Q: {scenario_question}")
        # print(f"A: {scenario_response}")
        
        # Shutdown agent
        print("\nShutting down agent...")
        await agent.shutdown()
    
    asyncio.run(demo()) 