from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Sequence
import os
import json

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    """
    A class for managing a vector database for credit card information.
    
    This class handles the storage and retrieval of credit card data using vector embeddings.
    It supports both JSON and PDF data sources, and provides methods for searching and
    retrieving card information.
    
    Attributes:
        db_path (str): Path to the vector database directory
        collection_name (str): Name of the collection in the vector database
        embedding_model: The embedding model used for vectorization
        vector_store: The vector store instance (e.g., Chroma)
    """
    
    def __init__(self, db_path: str, collection_name: str = "default", 
                embedding_model: Optional[Union[str, Any]] = None) -> None:
        """
        Initialize the VectorDB instance.
        
        Args:
            db_path (str): Path to the vector database directory
            collection_name (str, optional): Name of the collection. Defaults to "default"
            embedding_model (Optional[Union[str, Any]], optional): Embedding model to use.
                Can be a string identifier or a model instance. Defaults to None.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Set up the embedding model
        self.embedding_model = self._setup_embedding_model(embedding_model)
        
        # Initialize vector store
        self.vector_store = self._setup_vector_store()
    
    def _setup_embedding_model(self, embedding_model: Optional[Union[str, Any]]) -> Optional[Any]:
        """
        Set up the embedding model based on the provided configuration.
        
        Args:
            embedding_model (Optional[Union[str, Any]]): Embedding model configuration
            
        Returns:
            Optional[Any]: Configured embedding model or None if setup fails
        """
        if embedding_model is not None:
            if isinstance(embedding_model, str):
                return OpenAIEmbeddings(model=embedding_model)
            else:
                return embedding_model
        
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbeddings()
        
        return None
    
    def _setup_vector_store(self) -> Optional[Chroma]:
        """
        Set up the vector store using the configured embedding model.
        
        Returns:
            Optional[Chroma]: Initialized vector store or None if setup fails
        """
        if self.embedding_model is not None:
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.db_path
            )
        return None
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a single document to the vector store.
        
        Args:
            doc_id (str): Unique identifier for the document
            content (str): Content of the document
            metadata (Optional[Dict[str, Any]], optional): Additional metadata. Defaults to None
        """
        if self.vector_store is not None:
            # Add document to real vector store
            doc = Document(
                page_content=content,
                metadata={"id": doc_id, **(metadata or {})}
            )
            self.vector_store.add_documents([doc])
            return
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Add multiple documents to the vector store in batches.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to add
            batch_size (int, optional): Number of documents to process in each batch. Defaults to 100
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

    def search_cards(self, query: str) -> None:
        """
        Search for credit cards based on a query string.
        
        Args:
            query (str): Search query
        """
        print(f"Searching for: {query}")
        results = self.vector_store.similarity_search(query, k=10)
        print(f"Raw results count: {len(results)}")

        if results:
            temp = []
            for i, result in enumerate(results, 1):
                card_id = result.metadata.get('id')
                #base_card_id = card_id.split('_chunk')[0]   
                if card_id not in temp:
                    temp.append(card_id)

        if not temp:
            print("No matching cards found.")

        print("Top card IDs:")
        for i in range(len(temp)):
            print(f"{i+1}: {temp[i]}")
        return 
    
    def get_card_details(self, card_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a specific credit card.
        
        Args:
            card_id (str): ID of the card to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Card details if found, None otherwise
        """
        json_dir = f"data/card/json"
        
        # Call load_card_data with the specific directory
        documents, _ = load_card_data(json_dir)
        print(f"Loaded {len(documents)} documents from {json_dir}")
        matching_docs = [doc for doc in documents if doc['id'] == card_id]
        
        if matching_docs:
            doc = matching_docs[0] 
            return {
                'id': doc['id'],
                'content': doc['content'][:2000]
            }
        print(f"No details found for card ID: {card_id} in {json_dir}")
        return None
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform a similarity search in the vector store.
        
        Args:
            query (str): Search query
            top_k (int, optional): Number of top results to return. Defaults to 3
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores
        """
        if self.vector_store is not None:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            
            return [
                {
                    'id': doc.metadata.get('id', f"doc_{i}"),
                    'content': doc.page_content,
                    'metadata': {k: v for k, v in doc.metadata.items() if k != 'id'},
                    'score': score
                }
                for i, (doc, score) in enumerate(results)
            ]
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document from the vector store.
        
        Args:
            doc_id (str): ID of the document to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Document if found, None otherwise
        """
        if self.vector_store is not None:
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
    
    def create_rag_chain(self, llm: Optional[Any] = None, prompt: Optional[PromptTemplate] = None, 
                        search_type: str = "similarity", search_kwargs: Dict[str, Any] = {"k": 3}) -> Any:
        """
        Create a RAG chain for question answering.
        
        Args:
            llm (Optional[Any], optional): Language model to use. Defaults to None
            prompt (Optional[PromptTemplate], optional): Prompt template. Defaults to None
            search_type (str, optional): Type of search to perform. Defaults to "similarity"
            search_kwargs (Dict[str, Any], optional): Additional search parameters. Defaults to {"k": 3}
            
        Returns:
            Any: Configured RAG chain
            
        Raises:
            ValueError: If vector store is not properly initialized
        """
        if self.vector_store is None:
            raise ValueError("RAG chains require a real vector store with embeddings")
        
        # Set up LLM if not provided
        if llm is None:
            llm = ChatOpenAI(temperature=0)
        
        # Set up retriever with multi-query
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(
                search_type=search_type, 
                search_kwargs=search_kwargs
            ),
            llm=llm
        )
        
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
        
        def format_docs(docs: Sequence[Document]) -> str:
            """
            Format a sequence of documents into a single string.
            
            Args:
                docs (Sequence[Document]): Documents to format
                
            Returns:
                str: Formatted documents as a single string
            """
            formatted = "\n\n".join(doc.page_content for doc in docs)
            return formatted
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        return rag_chain

def load_card_data(json_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    documents = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    card_data_dict = {}
    
    for json_file in json_files:
        try:
            with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
                card_data = json.load(f)
                
                rewards_benefits = card_data.get('rewards_and_benefits', {})
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
                
                card_data_dict[card_data.get('card_name', '')] = card_data
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return documents, card_data_dict

def load_card_mapping(mapping_file: str) -> Dict[str, Any]:
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_pdf_data(pdf_dir: str, card_name_mapping: Dict[str, Any], card_data_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)

            # Get card names this PDF maps to
            mapped_names = card_name_mapping.get(pdf_file, [pdf_file.replace('.pdf', '')])
            if isinstance(mapped_names, str):
                mapped_names = [mapped_names]
            
            # Duplicate chunks for each mapped card
            for i, chunk in enumerate(chunks):
                if not chunk.page_content.strip():
                    continue

                for card_name in mapped_names:
                    doc_id = f"{card_name}_chunk_{i}"
                    metadata = {
                        "source": doc_id,
                        "page": chunk.metadata.get("page", 0),
                        "chunk": i,
                        "type": "pdf",
                        "card_name": card_name,
                        "card_type": card_data_dict.get(card_name, {}).get('card_type', ''),
                        "issuer": card_data_dict.get(card_name, {}).get('issuer', ''),
                        "card_association": card_data_dict.get(card_name, {}).get('card_association', '')
                    }
                    documents.append({
                        "id": card_name,
                        "content": chunk.page_content,
                        "metadata": metadata
                    })

        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    
    return documents


if __name__ == "__main__":
    # Initialize vector database
    db_path = "data/vector_db/cards"
    
    # Delete the database if it exists
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        
    db = VectorDB(db_path, collection_name="credit_cards", embedding_model="text-embedding-3-large")
    
    # Load card data from JSON files
    json_dir = "data/card/json"
    json_documents, card_data_dict = load_card_data(json_dir)
    
    # Load PDF data
    pdf_dir = "data/card/pdf"
    mapping_file = os.path.join(pdf_dir, '_mapping.json')

    card_name_mapping = load_card_mapping(mapping_file)
    pdf_documents = load_pdf_data(pdf_dir, card_name_mapping, card_data_dict)
    
    # Combine all documents
    all_documents = json_documents + pdf_documents
    
    # Add documents to vector database
    print(f"Adding {len(json_documents)} cards and {len(pdf_documents)} PDF chunks to vector database...")
    db.add_documents(all_documents)
    
    if os.environ.get("OPENAI_API_KEY"):
        print("\nTesting search_cards with ranked card names...")
        queries = ["air miles", "cashback on online shopping", "dining rewards"]

        for query in queries:
            results = db.search_cards(query)
            print(results)
        
        # Test get_card_details 
        print("\nTesting get_card_details...")
        card_id = "Citi PremierMiles Card"
        details = db.get_card_details(card_id)
        if details:
            print(f"Details for {card_id}:")
            print(f"Content: {details['content']}...")

        print("\nTesting RAG chain with questions...")
        rag_chain = db.create_rag_chain()
        
        questions = [
            "I want cards that have rewards for miles. Compare the two.",
            "Show me cashback cards from DBS and compare them.",
            "What are the best dining rewards cards? Compare three."
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = rag_chain.invoke(question)
            print(f"Answer: {response.content}") 