from mcp.server.fastmcp import FastMCP
import asyncio
import os
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

# Initialize the MCP server for card data access using FastMCP
server = FastMCP("card_data")

@server.tool()
async def get_available_cards() -> List[Dict[str, Any]]:
    """Returns list of all cards with basic metadata
    
    Returns:
        list: A list of dictionaries containing basic card information
    """
    

@server.tool()
async def get_card_details(card_id: str) -> Dict[str, Any]:
    """Returns full card information in its original format
    
    Args:
        card_id (str): The ID of the card to retrieve details for
        
    Returns:
        dict: Complete details of the requested card
    """
    # Dummy implementation - will be replaced with actual database/file lookup
    dummy_data = {
        "citi_premiermiles": {
            "id": "citi_premiermiles",
            "name": "Citi PremierMiles Card",
            "type": "miles",
            "annual_fee": 192.60,
            "miles_rate": {
                "local": 1.2,
                "overseas": 2.0,
                "special_categories": {"online_travel": 3.0}
            },
            "promotion": "20,000 bonus miles with $9,000 spend in first 3 months",
            "min_income": 30000,
            "benefits": ["Priority Pass membership", "Travel insurance", "Flexible miles redemption"],
            "terms_and_conditions": "Full terms and conditions would be here in a real implementation."
        },
        "dbs_altitude": {
            "id": "dbs_altitude",
            "name": "DBS Altitude Card",
            "type": "miles",
            "annual_fee": 192.60,
            "miles_rate": {
                "local": 1.2,
                "overseas": 2.0,
                "special_categories": {"online_spend": 3.0}
            },
            "promotion": "10,000 bonus miles upon approval",
            "min_income": 30000,
            "benefits": ["Complimentary travel insurance", "Gourmet Collection privileges"],
            "terms_and_conditions": "Full terms and conditions would be here in a real implementation."
        },
        "uob_one": {
            "id": "uob_one",
            "name": "UOB One Card",
            "type": "cashback",
            "annual_fee": 192.60,
            "cashback_rate": {
                "base": 0.03,
                "tier_spend": 2000,
                "tier_cashback": 0.05
            },
            "promotion": "Up to 10% cashback on selected categories",
            "min_income": 30000,
            "benefits": ["No minimum spend requirement", "Wide coverage of spending categories"],
            "terms_and_conditions": "Full terms and conditions would be here in a real implementation."
        },
        "ocbc_365": {
            "id": "ocbc_365",
            "name": "OCBC 365 Credit Card",
            "type": "cashback",
            "annual_fee": 192.60,
            "cashback_rate": {
                "dining": 0.06,
                "groceries": 0.03,
                "transport": 0.03,
                "utilities": 0.03
            },
            "promotion": "6% cashback on dining and online food delivery",
            "min_income": 30000,
            "benefits": ["Weekend dining deals", "Petrol discounts"],
            "terms_and_conditions": "Full terms and conditions would be here in a real implementation."
        },
        "amex_krisflyer": {
            "id": "amex_krisflyer",
            "name": "AMEX KrisFlyer Credit Card",
            "type": "miles",
            "annual_fee": 176.55,
            "miles_rate": {
                "local": 1.1,
                "overseas": 2.0,
                "singapore_airlines": 3.0
            },
            "promotion": "5,000 bonus KrisFlyer miles upon first spend",
            "min_income": 30000,
            "benefits": ["Complimentary travel insurance", "Special events access"],
            "terms_and_conditions": "Full terms and conditions would be here in a real implementation."
        }
    }
    
    # Return the requested card or an error message
    return dummy_data.get(card_id, {"error": f"Card with ID '{card_id}' not found"})

@server.tool()
async def query_tc(question: str, card_id: str) -> Dict[str, Any]:
    """Query the T&C database with natural language questions
    
    Args:
        question (str): The natural language question about terms and conditions
        card_id (str): The ID of the card to query about
        
    Returns:
        dict: Answer with source information and confidence score
    """
    # In a production implementation, this would use a RAG-based retrieval pipeline:
    # 1. Convert the question to an embedding vector
    # 2. Find the most similar T&C documents in the vector database
    # 3. Generate an answer based on the retrieved documents
    
    # Dummy implementation for demonstration
    dummy_responses = {
        "citi_premiermiles": {
            "annual fee": {
                "answer": "The annual fee for the Citi PremierMiles Card is S$192.60 (inclusive of GST).",
                "source": "Section 1.2 of Terms and Conditions",
                "confidence": 0.95
            },
            "miles expiry": {
                "answer": "Citi Miles do not expire.",
                "source": "Section 3.4 of Terms and Conditions",
                "confidence": 0.92
            },
            "minimum spend": {
                "answer": "There is no minimum spend requirement for earning base miles.",
                "source": "Section 2.1 of Terms and Conditions",
                "confidence": 0.88
            },
            "airport lounge": {
                "answer": "You get complimentary Priority Pass membership with 2 free visits per year.",
                "source": "Section 4.2 of Terms and Conditions",
                "confidence": 0.90
            }
        },
        "dbs_altitude": {
            "annual fee": {
                "answer": "The annual fee for the DBS Altitude Card is S$192.60 (inclusive of GST).",
                "source": "Section 1.3 of Terms and Conditions",
                "confidence": 0.95
            },
            "miles expiry": {
                "answer": "DBS miles do not expire as long as your card remains valid and in good standing.",
                "source": "Section 2.5 of Terms and Conditions",
                "confidence": 0.90
            }
        },
        "uob_one": {
            "annual fee": {
                "answer": "The annual fee for the UOB One Card is S$192.60 (inclusive of GST).",
                "source": "Section 1.1 of Terms and Conditions",
                "confidence": 0.95
            },
            "cashback cap": {
                "answer": "Cashback is capped at S$100 per month.",
                "source": "Section 2.3 of Terms and Conditions",
                "confidence": 0.93
            }
        }
    }
    
    # Simple keyword matching for the demo
    if card_id not in dummy_responses:
        return {
            "answer": f"I don't have information for the card with ID '{card_id}'.",
            "source": "N/A",
            "confidence": 0.0
        }
    
    # Very simple keyword matching for demonstration
    for keyword, response in dummy_responses[card_id].items():
        if keyword in question.lower():
            return response
    
    return {
        "answer": "I couldn't find specific information to answer that question.",
        "source": "N/A",
        "confidence": 0.0
    }

@server.tool()
async def search_cards(query: str) -> List[Dict[str, Any]]:
    """Semantic search for cards matching natural language criteria
    
    Args:
        query (str): Natural language query about card features 
                    (e.g., "high miles for dining", "no annual fee")
        
    Returns:
        list: A list of dictionaries containing matching cards with relevance scores
    """
    # In a production implementation, this would use a vector similarity search:
    # 1. Convert the query to an embedding vector
    # 2. Find cards with similar feature vectors in the vector database
    # 3. Return ranked results with similarity scores and explanations
    
    # Simple keyword matching for demonstration
    all_cards = await get_available_cards()
    results = []
    
    # Very basic keyword matching - will be replaced with actual semantic search
    query_lower = query.lower()
    
    # Match miles cards
    if "miles" in query_lower:
        for card in all_cards:
            if card["type"] == "miles":
                score = 0.8  # Arbitrary score
                reason = "Miles rewards card"
                
                if "dining" in query_lower and "premiermiles" in card["id"]:
                    score = 0.9
                    reason = "Good miles earning for dining expenses"
                    
                if "travel" in query_lower and "altitude" in card["id"]:
                    score = 0.85
                    reason = "Good for travel-related expenses"
                    
                if "singapore airlines" in query_lower and "krisflyer" in card["id"]:
                    score = 0.95
                    reason = "Direct KrisFlyer miles earning with Singapore Airlines"
                    
                results.append({
                    "card_id": card["id"],
                    "name": card["name"],
                    "relevance": score,
                    "match_reason": reason
                })
    
    # Match cashback cards
    elif "cashback" in query_lower or "cash back" in query_lower:
        for card in all_cards:
            if card["type"] == "cashback":
                score = 0.75  # Arbitrary score
                reason = "Cashback rewards card"
                
                if "dining" in query_lower and "365" in card["id"]:
                    score = 0.9
                    reason = "High cashback for dining expenses"
                    
                if "grocery" in query_lower and "365" in card["id"]:
                    score = 0.85
                    reason = "Good cashback for grocery shopping"
                    
                if "general" in query_lower and "one" in card["id"]:
                    score = 0.88
                    reason = "Good for general spending with tier-based cashback"
                    
                results.append({
                    "card_id": card["id"],
                    "name": card["name"],
                    "relevance": score,
                    "match_reason": reason
                })
    
    # Match by annual fee
    elif "annual fee" in query_lower or "no fee" in query_lower:
        limit = float('inf')
        if "no" in query_lower:
            limit = 0
        elif "low" in query_lower:
            limit = 100
        
        for card in all_cards:
            if card["annual_fee"] <= limit:
                results.append({
                    "card_id": card["id"],
                    "name": card["name"],
                    "relevance": 0.8,
                    "match_reason": f"Annual fee within limit (${card['annual_fee']:.2f})"
                })
    
    # Default behavior for other queries - return all with low relevance
    else:
        for card in all_cards:
            results.append({
                "card_id": card["id"],
                "name": card["name"],
                "relevance": 0.5,
                "match_reason": "General match"
            })
    
    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results

if __name__ == "__main__":
    # # Determine port from environment or use default
    # port = int(os.environ.get('PORT', 8001))
    
    # # Set the port in the environment for the server to use
    # os.environ['MCP_PORT'] = str(port)
    
    # print(f"Starting Card Data MCP Server on port {port}...")
    
    # Run with SSE transport
    server.run(transport='sse') 