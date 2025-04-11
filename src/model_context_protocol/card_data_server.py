from pathlib import Path
from mcp.server.fastmcp import FastMCP
import asyncio
import os
from typing import List, Dict, Any, Optional
import json
import random
import logging
import time
import traceback
import sys

# Add project root to Python path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_openai import OpenAIEmbeddings

from src.card_processing.vector_db import VectorDB

# Set up logging configuration
logger = logging.getLogger("card_data_server")
logger.setLevel(logging.DEBUG)

# Create a file handler
logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "logs"
logs_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(logs_dir / "card_data_server.log")
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize the MCP server for card data access using FastMCP
server = FastMCP("card_data")

db = VectorDB(
    db_path="data/vector_db/cards", 
    collection_name="credit_cards",
    embedding_model="text-embedding-3-large"
)


@server.tool()
async def get_available_cards() -> List[Dict[str, Any]]:
    """Returns list of all cards with basic metadata
    
    Returns:
        list: A list of dictionaries containing basic card information
    """
    logger.info("get_available_cards called")
    start_time = time.time()
    
    try:
        # Get all card details
        logger.debug("Retrieving all card metadata from vector store")
        metadata_start = time.time()
        metadatas = db.vector_store.get(include=["metadatas"])["metadatas"]
        metadata_time = time.time() - metadata_start
        logger.debug(f"Metadata retrieval took {metadata_time:.4f} seconds")
        
        unique_card_names = set(
            (card["card_name"], card["card_type"], card["issuer"], card["card_association"])
            for card in metadatas 
            if card and "card_name" in card
        )
        
        # Convert to list of dictionaries
        cards = [
            {
                "card_id": card_name,
                "card_type": card_type,
                "issuer": issuer,
                "card_association": card_association
            }
            for card_name, card_type, issuer, card_association in unique_card_names
        ]
        
        total_time = time.time() - start_time
        logger.info(f"get_available_cards completed in {total_time:.4f} seconds, found {len(cards)} cards")
        return cards
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"get_available_cards failed after {total_time:.4f} seconds: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


@server.tool()
async def get_card_details(card_id: str, json_dir: str = "data/card/json") -> Dict[str, Any]:
    """Returns full card information in its original format
    
    Args:
        card_id (str): The ID of the card to retrieve details for
        json_dir (str): The directory containing the JSON files
        
    Returns:
        dict: Complete details of the requested card
    """
    logger.info(f"get_card_details called for card: {card_id}")
    start_time = time.time()
    
    try:
        logger.debug(f"Loading card data from {os.path.join(json_dir, card_id)}.json")
        with open(os.path.join(json_dir, card_id) + '.json', 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        
        total_time = time.time() - start_time
        logger.info(f"get_card_details completed in {total_time:.4f} seconds")
        return card_data
    except FileNotFoundError:
        total_time = time.time() - start_time
        logger.error(f"Card with ID '{card_id}' not found after {total_time:.4f} seconds")
        return {'error': f"Card with ID '{card_id}' not found"}
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Error loading card with ID '{card_id}' after {total_time:.4f} seconds: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'error': f"Error loading card with ID '{card_id}': {e}"}

@server.tool()
async def query_tc(question: str, card_id: str, num_candidates: int = 5, llm_model: str = "gpt-3.5-turbo-0125") -> Dict[str, Any]:
    """Query the T&C database with natural language questions
    
    Args:
        question (str): The natural language question about terms and conditions
        card_id (str): The ID of the card to query about
        num_candidates (int): The number of candidates to return (default is 5)
        llm_model (str): The LLM model to use (default is "gpt-3.5-turbo-0125")
        
    Returns:
        dict: Answer with source information
    """
    logger.info(f"query_tc called with question: '{question}', card_id: {card_id}")
    start_time = time.time()
    
    try:
        rag_chain = db.create_rag_chain(
            llm=ChatOpenAI(model=llm_model) if llm_model else None,
            search_type="similarity",
            search_kwargs={
                "k": num_candidates,
                "filter": {"card_name": card_id}
            }
        )
        result = rag_chain.invoke(question)
        
        total_time = time.time() - start_time
        logger.info(f"query_tc completed in {total_time:.4f} seconds")
        return result.model_dump()
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"query_tc failed after {total_time:.4f} seconds: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error processing query: {str(e)}"

@server.tool()
async def search_cards(query: str, num_candidates: int = 10) -> List[Dict[str, Any]]:
    """Semantic search for cards matching natural language criteria
    
    Args:
        query (str): Natural language query about card features 
                    (e.g., "high miles for dining", "no annual fee")
        num_candidates (int): The number of candidates to return (default is 10)
        
    Returns:
        list: A list of dictionaries containing matching cards with relevance scores.
                Note: the distance score is euclidean distance, so lower is better.
    """
    logger.info(f"search_cards called with query: '{query}', num_candidates: {num_candidates}")
    start_time = time.time()
    
    try:
        # Limit the number of candidates to a reasonable range
        safe_num = min(max(1, num_candidates), 20)
        logger.debug(f"Using {safe_num} candidates (after safety limits)")
        
        results = db.search(query, top_k=safe_num)
        
        total_time = time.time() - start_time
        logger.info(f"search_cards completed in {total_time:.4f} seconds with {len(results)} results")
        return results
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"search_cards failed after {elapsed:.4f} seconds: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return an empty result to avoid breaking the conversation
        return [{'card_name': f'Error: {str(e)}', 'score': 0.0}]

@server.tool()
async def generate_questions(
    question_history: List[str] = [],
    num_questions: int = 5,
    llm_model: str = "gpt-3.5-turbo-0125"
    ) -> List[str]:
    """Generate questions based on the question history
    
    Args:
        question_history (List[str]): A list of previous questions.
                Default is an empty list.
        num_questions (int): The number of questions to generate.
                Default is 3.
        llm_model (str): The LLM model to use.
                Default is "gpt-3.5-turbo-0125".
        
    Returns:
        list: A list of questions
    """
    logger.info(f"generate_questions called with num_questions: {num_questions}")
    start_time = time.time()
    
    try:
        logger.debug(f"Setting up retriever and LLM with model: {llm_model}")
        setup_start = time.time()
        retriever = db.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,
            }
        )
        
        llm = ChatOpenAI(model=llm_model)
        prompt = hub.pull("rlm/rag-prompt")
        setup_time = time.time() - setup_start
        logger.debug(f"Retriever and LLM setup took {setup_time:.4f} seconds")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        logger.debug("Setting up RAG chain")
        chain_start = time.time()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        chain_time = time.time() - chain_start
        logger.debug(f"RAG chain setup took {chain_time:.4f} seconds")
        
        questions = []
        # Limit the number of questions for safety
        safe_num = min(max(1, num_questions), 10)
        logger.debug(f"Generating {safe_num} questions")
        
        for i in range(safe_num):
            logger.debug(f"Generating question {i+1}/{safe_num}")
            q_start = time.time()
            
            # Get available cards for cold-start questions
            if len(question_history) == 0:
                cards = await get_available_cards()
                random_card = random.choice(cards)
                random_card_id = random_card['card_id']
                logger.debug(f"Selected random card for question: {random_card_id}")
                prompt_text = f"Come up with a short, one-line question on {random_card_id} that can be answered by the following context."
            else:
                prompt_text = (
                    f"Come up with a short, one-line question.",
                    f"Additionally, make sure the question is relevant to all of these previously asked questions (but do not repeat an existing question): {questions}."
                )
                prompt_text = ' '.join(prompt_text)
            
            logger.debug(f"Using prompt: '{prompt_text}'")
            question = ""
            
            invoke_start = time.time()
            for chunk in rag_chain.stream(prompt_text):
                question += chunk
            invoke_time = time.time() - invoke_start
            logger.debug(f"Question generation took {invoke_time:.4f} seconds")
            
            if question:
                questions.append(question)
                logger.debug(f"Generated question: '{question}'")
            
            q_time = time.time() - q_start
            logger.debug(f"Question {i+1} generation completed in {q_time:.4f} seconds")
            
        total_time = time.time() - start_time
        logger.info(f"generate_questions completed in {total_time:.4f} seconds, generated {len(questions)} questions")
        return questions
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"generate_questions failed after {total_time:.4f} seconds: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [f"Error generating questions: {str(e)}"]

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Determine port from environment or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Set the port in the environment for the server to use
    os.environ['MCP_PORT'] = str(port)
    
    logger.info(f"Starting Card Data MCP Server on port {port}...")
    
    # Run with SSE transport
    server.run(transport='sse') 

    # ///////////////////////////
    # //        Notice         //
    # ///////////////////////////
    # The following test code requires the tests for vector_db.py to be run first.
    # In particular, the vector database must be created and have the same embedding 
    # model as the one used in the tests.

    async def run_tests():
        try:
            logger.info("Testing get_available_cards...")
            result = await get_available_cards()
            assert len(result) > 0
            logger.info(f"Found {len(result)} available cards")
            
            logger.info("Testing get_card_details...")
            result = await get_card_details("CIMB Visa Signature")
            assert len(result) > 0
            logger.info("Successfully retrieved card details")
            
            logger.info("Testing query_tc...")
            result = await query_tc("What is the annual fee for the CIMB Visa Signature card?", "CIMB Visa Signature")
            print(result, 'TEST', len(result), type(result))
            assert isinstance(result, dict) and len(result) > 0
            logger.info("Successfully queried terms and conditions")

            logger.info("Testing search_cards...")
            result = await search_cards("What card has the highest miles for dining?")
            assert isinstance(result, list) and len(result) > 0
            logger.info(f"Found {len(result)} cards matching the query")
            
            logger.info("Testing generate_questions...")
            result = await generate_questions()
            assert isinstance(result, list) and len(result) > 0
            logger.info(f"Generated {len(result)} questions")
            
            logger.info("All tests passed successfully!")
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Run the tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_tests())


   
