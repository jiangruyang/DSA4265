from typing import List, Dict, Any, Optional, Union, Callable
import os
import json
import asyncio
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Add multiple documents to the vector database
        
        Args:
            documents: List of document dictionaries, each with:
                      - id: Unique identifier
                      - content: Text content
                      - metadata: Optional metadata dict
            batch_size: Number of documents to process in each batch
        """
        if self.vector_store is not None:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                docs = [
                    Document(
                        page_content=doc["content"],
                        metadata={"id": doc["id"], **(doc.get("metadata", {}) or {})}
                    )
                        for doc in batch
                ]
                print(f"Processing batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
                self.vector_store.add_documents(docs)
            return

    def search_cards(self, query):
        print(f"Searching for: {query}")
        results = self.vector_store.similarity_search(query, k=10)
        print(f"Raw results count: {len(results)}")
        #print(results)


        if results:
                temp = []
                for i, result in enumerate(results, 1):
                    card_id = result.metadata.get('id')
                    base_card_id = card_id.split('_chunk')[0]   
                    if base_card_id not in temp:
                        temp.append(base_card_id)

        if not temp:
            print("No matching cards found.")

        print("Top card IDs:")
        for i in range(len(temp)):
            print(f"{i+1}: {temp[i]}")
        return 
    
    def get_card_details(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve card details by loading data from a deeper JSON directory
        
        Args:
            card_id: ID of the card (used as a subdirectory name)
            
        Returns:
            Dictionary with card details if found, None otherwise
        """
        # Construct the deeper directory path using card_id
        json_dir = f"data/card/json"
        
        
        # Call load_card_data with the specific directory
        documents = load_card_data(json_dir)
        print(f"Loaded {len(documents)} documents from {json_dir}")
        matching_docs = [doc for doc in documents if doc['id'] == card_id]
        
        # Assuming there's only one document per card_id directory
        if matching_docs:
            doc = matching_docs[0]  # Take the first document
            return {
                'id': doc['id'],
                'content': doc['content'][:2000]
                # 'metadata': doc['metadata']
            }
        print(f"No details found for card ID: {card_id} in {json_dir}")
        return None
        
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
        
        return None
    
    def create_rag_chain(self, llm=None, prompt=None):
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
            llm=llm
        )
        
        # Set up default RAG prompt if not provided
        if prompt is None:
            prompt = PromptTemplate.from_template(
                """Based on the provided context, identify and compare cards that match the user's query. 
                Focus on high-level features such as rewards type (e.g., miles, cashback), annual fees, 
                and key benefits. If the query specifies a number of cards to compare (e.g., "compare the two"), 
                limit the comparison to that number. Otherwise, provide a concise summary of all matching cards.
                Provide a structured comparison (as a table or bullet points) of cards relevant to the question.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:"""
            )
        
        # Format retrieved documents
        def format_docs(docs):
            formatted = "\n\n".join(doc.page_content for doc in docs)
            #print(formatted)
            return formatted
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        return rag_chain

def load_card_data(json_dir: str) -> List[Dict[str, Any]]:
    """Load card data from JSON files
    
    Args:
        json_dir: Directory containing card JSON files
        
    Returns:
        List of card documents ready for vector database
    """
    documents = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        try:
            with open(os.path.join(json_dir, json_file), 'r') as f:
                card_data = json.load(f)
                
                rewards_benefits = card_data.get('rewards_and_benefits', {})
                eligibility = card_data.get('eligibility', '')
                fees = card_data.get('fees', {})
                minimum_income = card_data.get('minimum_income', {})
                required_documents = card_data.get('documents_required', {})
                # Create a document from the card data
                doc_id = json_file.replace('.json', '')
                content = f"""
                Card Name: {card_data.get('card_name', '')}
                Card Type: {card_data.get('card_type', '')}
                Issuer: {card_data.get('issuer', '')}
                Key Features: {', '.join(rewards_benefits.get('key_features', []))}
                Air Miles: {', '.join(rewards_benefits.get('air_miles', []))}
                Overseas Spending: {', '.join(rewards_benefits.get('overseas_spending', []))}
                Petrol: {', '.join(rewards_benefits.get('petrol', []))}
                Rewards: {', '.join(rewards_benefits.get('rewards', []))}
                Buffet Promotion: {', '.join(rewards_benefits.get('buffet_promotion', []))}
                Dining: {', '.join(rewards_benefits.get('dining', []))}
                Online Shopping: {', '.join(rewards_benefits.get('online_shopping', []))}
                Shopping: {', '.join(rewards_benefits.get('shopping', []))}
                Installment: {', '.join(rewards_benefits.get('installment', []))}
                Entertainment: {', '.join(rewards_benefits.get('entertainment', []))}
                Grocery: {', '.join(rewards_benefits.get('grocery', []))}
                Cashback: {', '.join(rewards_benefits.get('cashback', []))}
                Bill Payment: {', '.join(rewards_benefits.get('bill_payment', []))}
                Student: {', '.join(rewards_benefits.get('student', []))}
                Eligibility: {', '.join(card_data.get('eligibility', []))}
                Annual Fees: {fees.get('annual_fee', '')}
                Supplementary Card Fee: {fees.get('supplementary_card_fee', '')}
                Annual Fee Waiver: {fees.get('annual_fee_waiver', '')}
                Interest Free Period: {fees.get('interest_free_period', '')}
                Annual Interest Rate: {fees.get('annual_interest_rate', '')}
                Late Payment Fee: {fees.get('late_payment_fee', '')}
                Minimum Monthly Repayment: {fees.get('minimum_monthly_repayment', '')}
                Foreign Transaction Fee: {fees.get('foreign_transaction_fee', '')}
                Cash Advance Fee: {fees.get('cash_advance_fee', '')}
                Overlimit Fee: {fees.get('overlimit_fee', '')}
                Minimum Singaporean/PR Income: {minimum_income.get('singaporean_pr', '')}
                Minimum Non-Singaporean Income: {minimum_income.get('non_singaporean', '')}
                Card Association: {card_data.get('card_association', '')}
                Singapore Citizen/PRs: {required_documents.get('sg_citizens_prs', '')}
                Passport Validity: {required_documents.get('passport_validity', '')}
                Work Permit Validity: {required_documents.get('work_permit_validity', '')}
                Utility Bill: {required_documents.get('utility_bill', '')}
                Income Tax Notice: {required_documents.get('income_tax_notice', '')}
                Latest Original Computerised Payslip: {required_documents.get('payslip', '')}
                """
                
                metadata = {
                    "card_name": card_data.get('card_name', ''),
                    "card_type": card_data.get('card_type', ''),
                    "issuer": card_data.get('issuer', ''),
                    "card_association": card_data.get('card_association', '')
                }
                
                documents.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata
                })
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return documents

def load_card_mapping(mapping_file: str) -> Dict[str, Any]:
    """Load card ID to name mapping from a JSON file"""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_pdf_data(pdf_dir: str, card_name_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load PDF data from PDF files
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of PDF documents ready for vector database
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            # Load PDF file
            pdf_path = os.path.join(pdf_dir, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split pages into chunks
            chunks = text_splitter.split_documents(pages)
            card_name = card_name_mapping.get(pdf_file, pdf_file.replace('.pdf', ''))
            
            # Create documents from chunks
            for i, chunk in enumerate(chunks):
                if not chunk.page_content.strip():
                    continue
                doc_id = f"{card_name}_chunk_{i}"
                content = chunk.page_content
                
                metadata = {
                    "source": doc_id,
                    "page": chunk.metadata.get("page", 0),
                    "chunk": i,
                    "type": "pdf"
                }
                
                documents.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata
                })
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    
    return documents

if __name__ == "__main__":
    # Initialize vector database
    db_path = "data/vector_db/cards"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    db = VectorDB(db_path, collection_name="credit_cards")
    
    # Load card data from JSON files
    json_dir = "data/card/json"
    json_documents = load_card_data(json_dir)
    
    # Load PDF data
    pdf_dir = "data/card/pdf"
    mapping_file = os.path.join(pdf_dir, '_mapping.json')
    card_name_mapping = load_card_mapping(mapping_file)
    pdf_documents = load_pdf_data(pdf_dir, card_name_mapping)
    
    # Combine all documents
    all_documents = json_documents + pdf_documents
    
    # Add documents to vector database
    print(f"Adding {len(json_documents)} cards and {len(pdf_documents)} PDF chunks to vector database...")
    db.add_documents(all_documents)
    
    # Example searches
    # print("\nExample searches:")
    # queries = [
    #     "What cards offer miles rewards?",
    #     "Show me DBS credit cards",
    #     "What are the terms and conditions for OCBC Rewards Card?"]
    
    # for query in queries:
    #     print(f"\nSearching for: {query}")
    #     results = db.search(query, top_k=3)
    #     for i, result in enumerate(results, 1):
    #         print(f"{i}. Score: {result['score']:.2f}")
    #         if result['metadata'].get('type') == 'pdf':
    #             print(f"   Source: {result['metadata']['source']}")
    #             print(f"   Page: {result['metadata']['page']}")
    #         else:
    #             print(f"   Card: {result['metadata']['card_name']}")
    #         print(f"   Content: {result['content'][:200]}...")
    
    # Test RAG chain if OpenAI API key is available
    if os.environ.get("OPENAI_API_KEY"):
        print("\nTesting search_cards with ranked card names...")
        #retriever = db.vector_store.as_retriever()
        queries = ["air miles", "cashback on online shopping", "dining rewards"]

        for query in queries:
            results = db.search_cards(query)
            print(results)
        
        # for query in queries:
        #     print(f"\nSearching for: {query}")
        #     results = db.search_cards(query)
            
        #     if not results:
        #         print("   No matching cards found.")
        #         continue
            
        #     print("   Top 5 most relevant cards:")
        #     for i, result in enumerate(results, 1):
        #         print(f"   {i}. {result['id']} (Score: {result['score']:.2f})")
            #print(results)
        
        # Test get_card_details (unchanged)
        print("\nTesting get_card_details...")
        card_id = "Citi PremierMiles Card"
        details = db.get_card_details(card_id)
        if details:
            print(f"Details for {card_id}:")
            print(f"Content: {details['content']}...")
            #print(f"Metadata: {details['metadata']}")

        print("\nTesting RAG chain with questions...")
        rag_chain = db.create_rag_chain()
        
        questions = [
            # "What are the benefits of the Citi PremierMiles Card?",
            # "What is the annual fee for the DBS Altitude Card?",
            # "What rewards do I get with the HSBC Revolution Card?",
            # "What are the terms and conditions for lounge access?",
            # "What are the key benefits mentioned in the PDF documents?",
            "I want cards that have rewards for miles. Compare the two.",
            "Show me cashback cards from DBS and compare them.",
            "What are the best dining rewards cards? Compare three."
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = rag_chain.invoke(question)
            print(f"Answer: {response.content}") 