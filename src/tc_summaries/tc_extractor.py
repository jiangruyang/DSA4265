"""
Module for extracting structured information from credit card T&C documents.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CardFeeStructure(BaseModel):
    annual_fee: float = Field(description="Annual fee in SGD")
    annual_fee_waiver_condition: str = Field(description="Condition for waiving annual fee, if any")
    supplementary_card_fee: float = Field(description="Annual fee for supplementary cards in SGD")
    late_payment_fee: float = Field(description="Late payment fee in SGD")
    foreign_transaction_fee_percentage: float = Field(description="Foreign transaction fee as percentage")
    cash_advance_fee: str = Field(description="Cash advance fee structure")

class RewardRate(BaseModel):
    category: str = Field(description="Spending category (e.g., Dining, Groceries, Travel)")
    rate: str = Field(description="Reward rate (e.g., '5% cashback', '4 miles per $1')")
    conditions: Optional[str] = Field(description="Any conditions for this rate")
    cap: Optional[str] = Field(description="Any cap on rewards for this category")

class CardBenefits(BaseModel):
    welcome_bonus: Optional[str] = Field(description="Welcome or sign-up bonus")
    welcome_bonus_conditions: Optional[str] = Field(description="Conditions to qualify for welcome bonus")
    lounge_access: Optional[str] = Field(description="Airport lounge access details")
    insurance_coverage: Optional[str] = Field(description="Insurance coverage details")
    other_benefits: List[str] = Field(description="Other notable card benefits")

class ImportantDisclaimers(BaseModel):
    minimum_income: str = Field(description="Minimum annual income requirement")
    minimum_spend: Optional[str] = Field(description="Minimum spend requirements, if any")
    reward_expiry: Optional[str] = Field(description="Expiry of rewards/miles/points")
    excluded_transactions: List[str] = Field(description="Categories excluded from earning rewards")
    other_limitations: List[str] = Field(description="Other important limitations or exclusions")

class CardSummary(BaseModel):
    card_name: str = Field(description="Name of the credit card")
    card_type: str = Field(description="Type of card (e.g., Cashback, Miles, Points)")
    issuer: str = Field(description="Issuing bank or institution")
    fees: CardFeeStructure = Field(description="Fee structure details")
    reward_rates: List[RewardRate] = Field(description="List of reward rates by category")
    benefits: CardBenefits = Field(description="Card benefits details")
    disclaimers: ImportantDisclaimers = Field(description="Important disclaimers and limitations")

class TCExtractor:
    """
    Class for extracting structured information from credit card T&C documents
    using LLMs and storing them in an easily accessible format.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o", 
                 temperature: float = 0.1,
                 output_dir: str = "./data/card_tcs/json"):
        """
        Initialize the TCExtractor.
        
        Args:
            model_name: Name of the LLM model to use.
            temperature: Temperature for LLM generation.
            output_dir: Directory to save extracted JSON summaries.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.output_dir = output_dir
        self.parser = PydanticOutputParser(pydantic_object=CardSummary)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial document analyst specializing in credit card terms and conditions.
Extract structured information from the credit card T&C document provided.
Focus on these key elements:
1. Basic card information (name, issuer, type)
2. Fee structure (annual fee, foreign transaction fees, etc.)
3. Reward rates for different spending categories
4. Benefits (welcome bonus, lounge access, etc.)
5. Important disclaimers and limitations

Format your response according to the specified schema.
Be precise and only include information explicitly stated in the document.
If information is not available, use null or empty lists as appropriate.
For numerical values like fees, extract just the number."""),
            ("human", "Here is the credit card T&C document to analyze:\n\n{document}"),
            ("human", "Please extract the structured information according to this schema:\n{format_instructions}")
        ])
        
    def extract_tc_summary(self, document: Document) -> CardSummary:
        """
        Extract a structured summary from a T&C document.
        
        Args:
            document: Document containing T&C text.
            
        Returns:
            Structured CardSummary object.
        """
        try:
            # Prepare the prompt
            prompt = self.extraction_prompt.format_messages(
                document=document.page_content[:15000],  # Limit to avoid token limits
                format_instructions=self.parser.get_format_instructions()
            )
            
            # Extract information using LLM
            response = self.llm.invoke(prompt)
            
            # Parse the response
            summary = self.parser.parse(response.content)
            
            # Ensure card name is set if provided in metadata
            if hasattr(document, 'metadata') and 'card_name' in document.metadata:
                summary.card_name = document.metadata['card_name']
                
            return summary
            
        except Exception as e:
            logger.error(f"Error extracting T&C summary: {str(e)}")
            raise
    
    def save_summary(self, summary: CardSummary) -> str:
        """
        Save a card summary to a JSON file.
        
        Args:
            summary: CardSummary object to save.
            
        Returns:
            Path to the saved file.
        """
        # Create a valid filename
        filename = f"{summary.card_name.lower().replace(' ', '_')}_summary.json"
        file_path = os.path.join(self.output_dir, filename)
        
        # Convert to dictionary
        summary_dict = summary.dict()
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2)
            
        logger.info(f"Saved card summary to {file_path}")
        return file_path
    
    def process_documents(self, documents: List[Document]) -> Dict[str, CardSummary]:
        """
        Process multiple T&C documents and extract summaries.
        
        Args:
            documents: List of documents to process.
            
        Returns:
            Dictionary mapping card names to their summaries.
        """
        summaries = {}
        
        for doc in documents:
            try:
                logger.info(f"Processing document for {doc.metadata.get('card_name', 'unknown card')}")
                summary = self.extract_tc_summary(doc)
                self.save_summary(summary)
                summaries[summary.card_name] = summary
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        
        return summaries
    
    def load_existing_summaries(self) -> Dict[str, CardSummary]:
        """
        Load existing JSON summaries from the output directory.
        
        Returns:
            Dictionary mapping card names to their summaries.
        """
        summaries = {}
        
        if not os.path.exists(self.output_dir):
            return summaries
            
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.output_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summary = CardSummary.parse_obj(data)
                        summaries[summary.card_name] = summary
                except Exception as e:
                    logger.error(f"Error loading summary from {filename}: {str(e)}")
        
        return summaries 