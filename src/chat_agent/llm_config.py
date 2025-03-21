import os
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Configure logging
logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.0, model_name: Optional[str] = None) -> ChatOpenAI:
    """
    Initialize and return a ChatOpenAI LLM instance.
    
    Args:
        temperature: Controls randomness in responses. Higher values (e.g., 0.8) make responses
                     more varied, lower values (e.g., 0.2) make them more deterministic.
        model_name: The OpenAI model to use. If None, defaults to environment variable.
    
    Returns:
        A configured ChatOpenAI instance.
    """
    if model_name is None:
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Log OpenAI configuration for debugging
    api_key = os.getenv("OPENAI_API_KEY", "")
    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "Not Set"
    logger.debug(f"Initializing OpenAI with model={model_name}, temperature={temperature}, API key={masked_key}")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )

def format_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    """
    Convert a list of role/content message dicts to LangChain message objects.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
    
    Returns:
        List of LangChain message objects.
    """
    formatted_messages = []
    
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(SystemMessage(content=message["content"]))
        elif message["role"] == "user":
            formatted_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(AIMessage(content=message["content"]))
    
    return formatted_messages

def get_synergy_prompt() -> str:
    """
    Return the system prompt for the card synergy recommendation.
    
    Returns:
        String containing the system prompt.
    """
    return """You are an expert credit card rewards optimizer for Singapore.
Your task is to analyze user spending patterns and preferences to recommend
the optimal combination of credit cards that will maximize their rewards.

For each recommendation:
1. Consider the user's spending categories and amounts
2. Factor in their preferences (miles vs. cashback)
3. Respect their annual fee tolerance
4. Provide a clear usage strategy explaining which card to use for which category

Be specific about reward rates, caps, and any minimum spend requirements.
Prioritize practical advice that the user can easily follow to maximize their rewards.
"""

def get_chat_prompt() -> str:
    """
    Return the system prompt for the T&C chat assistant.
    
    Returns:
        String containing the system prompt.
    """
    return """You are a helpful credit card assistant for Singapore credit cards.
Answer user questions about credit card Terms & Conditions, scenario planning,
and recommendation clarifications.

When answering:
1. Use facts from the credit card T&Cs when possible
2. If asked about a change in spending patterns, recalculate rewards accordingly
3. Always explain the source of your information (e.g., "According to DBS Altitude T&C...")
4. Be honest when you don't know an answer

Avoid speculating about conditions or details not in the T&Cs.
Be precise about reward rates, caps, exclusions, and qualification criteria.
""" 