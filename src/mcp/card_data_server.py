from mcp.server.fastmcp import FastMCP
from mcp import types
import asyncio
import os
from typing import List, Dict, Any, Optional, Iterator
import json
import random

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_openai import OpenAIEmbeddings

# Initialize the MCP server for card data access using FastMCP
server = FastMCP("card_data")

class VectorDBCache:
    def __init__(self, db_path: str = "data/vector_db/cards"):
        self.db_path = db_path
        self._db = None  # Use _db internally

    @property
    def db(self):
        if self._db is None:
            # Lazy-load Chroma
            self._db = Chroma(
                collection_name="credit_cards", 
                persist_directory=self.db_path, 
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
            )
        return self._db

cache = VectorDBCache()


@server.tool()
async def get_available_cards() -> List[Dict[str, Any]]:
    """Returns list of all cards with basic metadata
    
    Returns:
        list: A list of dictionaries containing basic card information
    """
    db = cache.db
    
    # Get all card details
    unique_card_names = set(
        (card["card_name"], card["card_type"] , card["issuer"], card["card_association"])
        for card in db.get(include=["metadatas"])["metadatas"] 
        if card and "card_name" in card
    )
    
    # Convert to list of dictionaries
    return [
        {
            "card_name": card_name,
            "card_type": card_type,
            "issuer": issuer,
            "card_association": card_association
        }
        for card_name, card_type, issuer, card_association in unique_card_names
    ]


@server.tool()
async def get_card_details(card_id: str, json_dir: str = "data/card/json") -> Dict[str, Any]:
    """Returns full card information in its original format
    
    Args:
        card_id (str): The ID of the card to retrieve details for
        json_dir (str): The directory containing the JSON files
        
    Returns:
        dict: Complete details of the requested card
    """
    try:
        with open(os.path.join(json_dir, card_id) + '.json', 'r', encoding='utf-8') as f:
            card_data = json.load(f)
    except FileNotFoundError:
        return {'error': f"Card with ID '{card_id}' not found"}
    except Exception as e:
        return {'error': f"Error loading card with ID '{card_id}': {e}"}
    
    return card_data

@server.tool()
async def query_tc(question: str, card_id: str, num_candidates: int = 5, llm_model: str = "gpt-3.5-turbo-0125") -> Dict[str, Any]:
    """Query the T&C database with natural language questions
    
    Args:
        question (str): The natural language question about terms and conditions
        card_id (str): The ID of the card to query about
        num_candidates (int): The number of candidates to return (default is 5)
        llm_model (str): The LLM model to use (default is "gpt-3.5-turbo-0125")
        
    Returns:
        dict: Answer with source information and confidence score
    """
    llm = ChatOpenAI(model=llm_model)
    prompt = hub.pull("rlm/rag-prompt")
    retriever = cache.db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": num_candidates,
            "filter": {"card_name": card_id}
        }
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)
        
        

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
    results = cache.db.similarity_search_with_score(query, k=num_candidates)
    return [
        {
            'card_name': result[0].metadata['card_name'],
            'score': result[1]
        }
        for result in results
    ]

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
    
    retriever = cache.db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
        }
    )
    
    llm = ChatOpenAI(model=llm_model)
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    questions = []
    for _ in range(num_questions):
        prompt = (
            # Cold-start question
            f"Come up with a short, one-line question on {random.choice(await get_available_cards())} that can be answered by the following context.",
        ) if len(question_history) == 0 else (
            # Chat-history-question
            f"Come up with a short, one-line question.",
            f"Additionally, make sure the question is relevant to all of these previously asked questions (but do not repeat an existing question): {questions}."
        )

        question = ""
        for chunk in rag_chain.stream(' '.join(prompt)):
            question += chunk

        questions.append(question)
            
    return questions


def iter_card_detail_resources() -> Iterator[types.Resource]:
    '''
    Iterates over all available card detail resources.
    
    Yields:
        types.Resource: A card detail resource
    '''
    # FastMCP does not expose `@app.list_resources()` like low-level MCP.
    # https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/fastmcp/server.py
    # So I'm doing this as a workaround.
    available_files = os.listdir("data/card/json")
    for file in available_files:
        resource = types.Resource(
            uri=f"file:///cards/{file}",
            name=file,
            mime_type="application/json",
        )
        yield resource


@server.tool()
async def list_card_detail_resources() -> List[str]:
    '''
    Lists all available card detail resources.
    '''
    return [file for file in os.listdir("data/card/json")]


@server.tool()
async def load_card_detail_resource(files: List[str]) -> None:
    '''
    Loads a card detail resource into the server from a list of file names.
    
    Example:
        load_card_detail_resource(["CIMB Visa Signature.json"])
    
    Args:
        files (List[str]): The list of file names of the card detail resources to load
    '''
    for file in files:
        resource = types.Resource(
            uri=f"file:///cards/{file}",
            name=file,
            mime_type="application/json",
        )
        server.add_resource(resource)


def load_all_card_detail_resources() -> None:
    '''
    Loads all available card detail resources into the server. I suspect this will be quite
    cost-intensive, so I'm not making it a tool yet.
    
    See `iter_card_detail_resources` for more details.
    '''
    for resource in iter_card_detail_resources():
        server.add_resource(resource)

    
@server.resource("{uri}")
async def get_card_detail_resource_uri(uri: str) -> dict:
    '''
    MCP Resource that reads a card detail resource from a URI.
    
    Args:
        uri (str): The URI of the card detail resource to read
        
    Returns:
        dict: The card detail json
    '''
    available_files = os.listdir("data/card/json")
    card_id = uri.split(":///cards/")[1]

    if card_id in available_files:
        with open(os.path.join("data/card/json", card_id), "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Resource not found")
    

@server.resource("file:///cards/{file}")
async def get_card_detail_resource(file: str) -> dict:
    '''
    MCP Resource that gets a card detail resource. Thin wrapper to 
    `get_card_detail_resource_uri` that takes a file name as input.
    
    Args:
        file (str): The file name of the card detail resource to get
        
    Returns:
        dict: The card detail json
    '''
    return await get_card_detail_resource_uri(f"file:///cards/{file}")

    
if __name__ == "__main__":
    # Determine port from environment or use default
    port = int(os.environ.get('PORT', 8001))
    
    # Set the port in the environment for the server to use
    os.environ['MCP_PORT'] = str(port)
    
    print(f"Starting Card Data MCP Server on port {port}...")
    
    # Run with SSE transport
    server.run(transport='sse') 

    # ///////////////////////////
    # //        Notice         //
    # ///////////////////////////
    # The following test code requires the tests for vector_db.py to be run first.
    # In particular, the vector database must be created and have the same embedding 
    # model as the one used in the tests.

    result = asyncio.run(get_available_cards())
    assert len(result) > 0
    
    result = asyncio.run(get_card_details("CIMB Visa Signature"))
    assert len(result) > 0
    
    result= asyncio.run(query_tc("What is the annual fee for the CIMB Visa Signature card?", "CIMB Visa Signature"))
    assert len(result) > 0

    result = asyncio.run(search_cards("What card has the highest miles for dining?"))
    assert len(result) > 0
    
    result = asyncio.run(generate_questions())
    assert len(result) > 0
 
    result = asyncio.run(server.list_resources())
    assert len(result) == 0

    async def test():
        await load_card_detail_resource("CIMB Visa Signature.json")
        result = asyncio.run(server.list_resources())
        assert len(result) > 0
        
        result = asyncio.run(get_card_detail_resource("CIMB Visa Signature.json"))
        assert result is not None
        
    asyncio.run(test())
        
 
 
    
