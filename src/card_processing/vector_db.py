from typing import List, Dict, Any, Optional, Union, Callable
import os
import json
import asyncio
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

class VectorDB:
    """Vector database for storing and retrieving embeddings
    
    This class provides a simple interface for storing and retrieving
    document embeddings. It's used for both card embeddings and 
    Terms & Conditions (T&C) document embeddings, with Langchain 
    integration for improved RAG capabilities.
    """
    
    def __init__(self, db_path: str, collection_name: str = "default", 
                embedding_model: Optional[Union[str, Any]] = None):
        """Initialize the vector database
        
        Args:
            db_path: Path to the vector database directory
            collection_name: Name of the collection to use
            embedding_model: Embedding model to use, either a string identifier
                           or a Langchain embedding model instance
                           (if None, uses OpenAIEmbeddings if API key set,
                           otherwise falls back to a dummy implementation)
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Set up the embedding model
        self.embedding_model = self._setup_embedding_model(embedding_model)
        
        # Initialize vector store
        self.vector_store = self._setup_vector_store()
        
        # For fallback with dummy implementation
        self.documents = {}
        self.embeddings = {}
        
        # Load existing data for dummy implementation if needed
        self._load_dummy_database()
    
    def _setup_embedding_model(self, embedding_model):
        """Set up the embedding model to use"""
        if embedding_model is not None:
            if isinstance(embedding_model, str):
                # Future: Handle different embedding model types based on string ID
                return OpenAIEmbeddings(model=embedding_model)
            else:
                # Assume it's already a Langchain embedding model
                return embedding_model
        
        # Check if OpenAI API key is available
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbeddings()
        
        # Fallback to dummy implementation
        print("Warning: No embedding model provided and no OpenAI API key found.")
        print("Using dummy implementation for vector database.")
        return None
    
    def _setup_vector_store(self):
        """Set up the vector store based on embedding model availability"""
        if self.embedding_model is not None:
            # Use Chroma with real embedding model
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.db_path
            )
        
        # Return None for dummy implementation
        return None
    
    def _load_dummy_database(self):
        """Load existing database from disk for dummy implementation"""
        if self.vector_store is not None:
            # Using real vector store, no need for dummy data
            return
            
        index_path = os.path.join(self.db_path, f"{self.collection_name}_index.json")
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', {})
                    self.embeddings = data.get('embeddings', {})
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading vector database: {e}")
                # Initialize with empty data
                self.documents = {}
                self.embeddings = {}
    
    def _save_dummy_database(self):
        """Save current database to disk for dummy implementation"""
        if self.vector_store is not None:
            # Using real vector store, no need to save dummy data
            return
            
        index_path = os.path.join(self.db_path, f"{self.collection_name}_index.json")
        
        try:
            with open(index_path, 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f, indent=2)
        except IOError as e:
            print(f"Error saving vector database: {e}")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the vector database
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Additional metadata for the document
        """
        if self.vector_store is not None:
            # Add document to real vector store
            doc = Document(
                page_content=content,
                metadata={"id": doc_id, **(metadata or {})}
            )
            self.vector_store.add_documents([doc])
            return
        
        # Dummy implementation
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {}
        }
        
        # Create dummy embedding
        self.embeddings[doc_id] = [0.1, 0.2, 0.3, 0.4, 0.5]  # Dummy vector
        
        # Save changes to disk
        self._save_dummy_database()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents to the vector database
        
        Args:
            documents: List of document dictionaries, each with:
                      - id: Unique identifier
                      - content: Text content
                      - metadata: Optional metadata dict
        """
        if self.vector_store is not None:
            # Add documents to real vector store
            docs = [
                Document(
                    page_content=doc["content"],
                    metadata={"id": doc["id"], **(doc.get("metadata", {}) or {})}
                )
                for doc in documents
            ]
            self.vector_store.add_documents(docs)
            return
        
        # Dummy implementation
        for doc in documents:
            doc_id = doc["id"]
            self.documents[doc_id] = {
                'content': doc["content"],
                'metadata': doc.get("metadata", {}) or {}
            }
            
            # Create dummy embedding
            self.embeddings[doc_id] = [0.1, 0.2, 0.3, 0.4, 0.5]  # Dummy vector
        
        # Save changes to disk
        self._save_dummy_database()
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity
        
        Args:
            query: Text query to search for
            top_k: Number of top results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        if self.vector_store is not None:
            # Use real vector store for search
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            
            # Format results
            return [
                {
                    'id': doc.metadata.get('id', f"doc_{i}"),
                    'content': doc.page_content,
                    'metadata': {k: v for k, v in doc.metadata.items() if k != 'id'},
                    'score': score
                }
                for i, (doc, score) in enumerate(results)
            ]
        
        # Dummy implementation with keyword search
        query_lower = query.lower()
        results = []
        
        for doc_id, doc in self.documents.items():
            content = doc['content'].lower()
            metadata = doc['metadata']
            
            # Simple keyword matching
            score = 0.0
            for word in query_lower.split():
                if word in content:
                    score += 0.2  # Increase score for each matching word
            
            # Boost score based on metadata matches
            if metadata.get('card_id') and metadata.get('card_id') in query_lower:
                score += 0.5
                
            if metadata.get('section') and metadata.get('section').lower() in query_lower:
                score += 0.3
            
            # Add to results if there's any match
            if score > 0:
                results.append({
                    'id': doc_id,
                    'content': doc['content'],
                    'metadata': metadata,
                    'score': min(score, 1.0)  # Cap score at 1.0
                })
        
        # Sort by score (descending) and take top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document dict if found, None otherwise
        """
        if self.vector_store is not None:
            # For real vector store, we need to search by metadata
            results = self.vector_store.get(
                where={"id": doc_id}
            )
            
            if results and len(results) > 0:
                doc = results[0]
                return {
                    'id': doc_id,
                    'content': doc.page_content,
                    'metadata': {k: v for k, v in doc.metadata.items() if k != 'id'}
                }
            return None
        
        # Dummy implementation
        if doc_id in self.documents:
            return {
                'id': doc_id,
                'content': self.documents[doc_id]['content'],
                'metadata': self.documents[doc_id]['metadata']
            }
        return None
    
    def create_rag_chain(self, llm=None, prompt=None, num_queries: int = 3):
        """Create a RAG chain for question answering
        
        Args:
            llm: Language model to use for RAG (defaults to ChatOpenAI)
            prompt: Prompt template to use (defaults to a standard RAG prompt)
            num_queries: Number of query variations to generate for retrieval
            
        Returns:
            A callable chain that takes a question and returns an answer
        """
        if self.vector_store is None:
            raise ValueError("RAG chains require a real vector store with embeddings")
        
        # Set up LLM if not provided
        if llm is None:
            llm = ChatOpenAI(temperature=0)
        
        # Set up retriever with multi-query
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            llm=llm,
            prompt_key="multi_query_prompt",
            parser_key="multi_query_parser"
        )
        
        # Set up default RAG prompt if not provided
        if prompt is None:
            prompt = PromptTemplate.from_template(
                """Answer the following question based only on the provided context:
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:"""
            )
        
        # Format retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        return rag_chain


# Simple demo of the vector database
if __name__ == "__main__":
    # Check if running with real embeddings or dummy mode
    use_real = os.environ.get("OPENAI_API_KEY") is not None
    mode = "real embeddings" if use_real else "dummy mode"
    
    print(f"Running vector database demo with {mode}")
    
    # Create vector database instance
    db_path = "../data/vector_db/tc_documents"
    db = VectorDB(db_path, collection_name="card_tc")
    
    # Add some sample T&C documents
    db.add_document(
        "citi_premiermiles_annual_fee",
        "The annual fee for the Citi PremierMiles Card is S$192.60 (inclusive of GST).",
        {
            "card_id": "citi_premiermiles",
            "section": "Annual Fee",
            "source": "Section 1.2 of Terms and Conditions"
        }
    )
    
    db.add_document(
        "citi_premiermiles_miles_expiry",
        "Citi Miles earned by the Citi PremierMiles Card do not expire.",
        {
            "card_id": "citi_premiermiles",
            "section": "Miles Expiry",
            "source": "Section 3.4 of Terms and Conditions"
        }
    )
    
    db.add_document(
        "dbs_altitude_annual_fee",
        "The annual fee for the DBS Altitude Card is S$192.60 (inclusive of GST).",
        {
            "card_id": "dbs_altitude",
            "section": "Annual Fee",
            "source": "Section 2.1 of Terms and Conditions"
        }
    )
    
    # Add document with multiple paragraphs
    db.add_document(
        "citi_premiermiles_lounge_access",
        """Citi PremierMiles Visa cardmembers receive complimentary Priority Pass membership.
        This includes 2 free visits per calendar year to participating airport lounges worldwide.
        Additional visits are charged at US$32 per person per visit.
        To access the lounge, you must present your Priority Pass membership card.""",
        {
            "card_id": "citi_premiermiles",
            "section": "Airport Lounge Access",
            "source": "Section 4.2 of Terms and Conditions"
        }
    )
    
    # Test search functionality
    print("Searching for: 'annual fee citi'")
    results = db.search("annual fee citi")
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.2f}")
        print(f"   Document: {result['content']}")
        print(f"   Metadata: {result['metadata']}")
        print()
    
    # Test RAG chain if using real embeddings
    if use_real:
        print("\nTesting RAG chain")
        print("Creating RAG chain for question answering")
        
        rag_chain = db.create_rag_chain()
        
        questions = [
            "What is the annual fee for the Citi PremierMiles card?",
            "Do miles expire with the Citi card?",
            "What lounge access do I get with Citi PremierMiles?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = rag_chain.invoke(question)
            print(f"Answer: {answer.content}") 