import os
import glob
from typing import List, Dict, Any, Optional
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

class DocumentLoader:
    """
    A class for loading and embedding T&C documents for the RAG pipeline.
    """
    
    def __init__(self, vector_db_path: Optional[str] = None, vector_db_type: str = "chroma"):
        """
        Initialize the DocumentLoader.
        
        Args:
            vector_db_path: Path to store the vector database. If None, uses environment variable.
            vector_db_type: Type of vector database to use ('chroma' or 'faiss').
        """
        self.vector_db_path = vector_db_path or os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.vector_db_type = vector_db_type.lower()
        self.embeddings = OpenAIEmbeddings()
        
        # Create the vector database directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
    
    def load_tc_documents(self, directory_path: str) -> List[Document]:
        """
        Load T&C documents from a directory.
        
        Args:
            directory_path: Path to the directory containing T&C documents.
        
        Returns:
            List of loaded documents.
        """
        documents = []
        
        # Find all text and PDF files in the directory
        file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
        file_paths.extend(glob.glob(os.path.join(directory_path, "*.pdf")))
        
        for file_path in file_paths:
            try:
                # Extract card name from filename
                card_name = self._extract_card_name(file_path)
                
                # Read file content
                if file_path.endswith(".pdf"):
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                else:  # .txt file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "card_name": card_name
                    }
                )
                
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def _extract_card_name(self, file_path: str) -> str:
        """
        Extract the card name from a file path.
        
        Args:
            file_path: Path to the T&C document file.
        
        Returns:
            Extracted card name.
        """
        # Get filename without extension
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        
        # Clean up the name (replace underscores with spaces, etc.)
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Try to extract card name using patterns
        patterns = [
            r'(.*?)\s*terms\s*and\s*conditions',
            r'(.*?)\s*t&c',
            r'(.*?)\s*tc'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the cleaned filename
        return name.strip()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for embedding.
        
        Args:
            documents: List of documents to split.
        
        Returns:
            List of document chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Document]) -> Any:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to embed.
        
        Returns:
            Vector store instance.
        """
        if self.vector_db_type == "faiss":
            return FAISS.from_documents(documents, self.embeddings)
        else:  # Default to Chroma
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.vector_db_path
            )
    
    def load_vector_store(self) -> Any:
        """
        Load an existing vector store.
        
        Returns:
            Vector store instance.
        """
        if self.vector_db_type == "faiss":
            return FAISS.load_local(
                folder_path=self.vector_db_path,
                embeddings=self.embeddings
            )
        else:  # Default to Chroma
            return Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
    
    def process_and_load(self, directory_path: str, force_reload: bool = False) -> Any:
        """
        Process T&C documents and load them into a vector store.
        
        Args:
            directory_path: Path to the directory containing T&C documents.
            force_reload: Whether to force reloading documents even if a vector store exists.
        
        Returns:
            Vector store instance.
        """
        # Check if vector store already exists
        vector_store_exists = os.path.exists(self.vector_db_path) and len(os.listdir(self.vector_db_path)) > 0
        
        if vector_store_exists and not force_reload:
            print("Loading existing vector store...")
            return self.load_vector_store()
        
        print("Processing documents and creating new vector store...")
        documents = self.load_tc_documents(directory_path)
        split_docs = self.split_documents(documents)
        
        if not split_docs:
            raise ValueError(f"No documents found in {directory_path}")
        
        vector_store = self.create_vector_store(split_docs)
        
        # Persist if using Chroma
        if self.vector_db_type != "faiss":
            vector_store.persist()
        
        return vector_store 