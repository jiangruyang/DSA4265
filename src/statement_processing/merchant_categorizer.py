import os
import json
import re
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY in .env file.")

# For inference only
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MerchantCategorizer:
    """A merchant categorization model that uses BERT for inference."""
    
    # Define categories as a class variable
    CATEGORIES = [
            "groceries",
            "dining",
            "transportation",
            "shopping",
            "entertainment",
            "travel",
            "utilities",
            "healthcare",
            "education",
            "others"
        ]
        
    def __init__(self, model_path: str = "models/merchant_categorizer"):
        """Initialize the merchant categorizer model
        
        Args:
            model_path: Path to the trained BERT model
        """
        self.categories = self.CATEGORIES.copy()  # Instance copy of categories
        self.confidence_threshold = 0.0  # Lowered confidence threshold for better categorization
        self._llm = None
        self._embed_model = None
        
        # Location indicators to remove from merchant names
        self.location_indicators = [
            # Common location abbreviations
            "si ng", "sgp", "sg", "spore",
            
            # Country and city names
            "singapore", "singapura",
            
            # Common districts and areas
            "kent ridge", "utown", "one north", "clementi", "jurong", "tampines",
            "bedok", "orchard", "novena", "bishan", "ang mo kio", "toa payoh",
            "serangoon", "hougang", "sengkang", "punggol", "woodlands", "yishun",
            "sembawang", "changi", "pasir ris", "kallang", "aljunied", "geylang",
            
            # Shopping malls and landmarks
            "one raffles", "raffles place", "jem", "somerset", "ion", "nex",
            "vivo", "tampines mall", "bedok mall", "jurong point", "plaza singapura",
            "nex", "lot one", "causeway point", "northpoint", "compass one",
            
            # Months (to remove date references)
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            
            # Common business suffixes
            "pte", "ltd", "inc", "llp", "plc", "corp", "corporation",
            
            # Building types
            "tower", "building", "centre", "center", "mall", "plaza", "hub",
            
            # Floor indicators
            "b1", "b2", "b3", "l1", "l2", "l3", "level 1", "level 2", "level 3"
        ]

        self.brand_substrings = json.load(open("data/categorization/brand_substrings.json"))
        
        # Common merchant name patterns to clean
        self.merchant_patterns = [
            (r'\d+', ''),  # Remove numbers
            (r'[^\w\s-]', ''),  # Remove special characters
            (r'\s+', ' '),  # Normalize whitespace
            (r'\s+\([^)]*\)', ''),  # Remove parenthetical text
            (r'\s*-\s*[^-]*$', ''),  # Remove text after last hyphen
            (r'\s*:\s*[^:]*$', ''),  # Remove text after last colon
        ]
        
        # Initialize tokenizer and model
        if os.path.exists(model_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.use_bert = True
                logger.info(f"Loaded BERT model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading BERT model: {str(e)}")
                raise
        else:
            raise ValueError(f"BERT model not found at {model_path}")
    
    
    def _clean_merchant_name(self, merchant_name: str) -> str:
        """Clean merchant name by removing location indicators and standardizing format
        
        Args:
            merchant_name: Raw merchant name
            
        Returns:
            Cleaned merchant name
        """
        if not merchant_name:
            return ""
            
        # Convert to lowercase
        cleaned = merchant_name.lower()
        
        # Remove date patterns (DD MMM, DD-MM-YYYY, etc.)
        cleaned = re.sub(r'\d{1,2}(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d{2}(nov|dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct)', '', cleaned, flags=re.IGNORECASE)
        
        # Remove location indicators (conservative approach)
        for indicator in self.location_indicators:
            cleaned = cleaned.replace(indicator, "")
            
        # Remove numbers (except keep single digits that might be part of the name)
        cleaned = re.sub(r'\d{2,}', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove trailing reference numbers
        cleaned = re.sub(r'ref\s+\d+\s*$', '', cleaned)
        
        return cleaned.strip()
    
    def _categorize_brand_hardcoded(self, merchant_name: str) -> str:
        """Categorize a merchant name using hardcoded rules"""
        for category, substrings in self.brand_substrings.items():
            for substring in substrings:
                if substring in merchant_name:
                    return category
        return None
    
    def _is_incoming_payment(self, merchant_name: str) -> bool:
        """Check if the transaction is an incoming payment that should be filtered out
        
        Args:
            merchant_name: Raw merchant name
            
        Returns:
            True if this is an incoming payment, False otherwise
        """
        # Common incoming payment indicators
        incoming_indicators = [
            "incoming", "received", "inward", "transfer from", 
            "deposit", "credited", "salary", "received",
            "inward remittance", "credit transfer"
        ]
        
        # Check if merchant name contains any incoming payment indicators
        merchant_lower = merchant_name.lower()
        for indicator in incoming_indicators:
            if indicator in merchant_lower:
                return True
                
        return False
    
    def _is_page_number_or_noise(self, merchant_name: str) -> bool:
        """Check if the merchant name is likely a page number or other noise
        
        Args:
            merchant_name: Raw merchant name
            
        Returns:
            True if this appears to be a page number or noise, False otherwise
        """
        # Check for repeated numbers or characters (like "4 4 4 4 4")
        if re.match(r'^(\d+\s+)+\d+$', merchant_name):
            return True
            
        # Check for standalone numbers that might be page numbers
        if re.match(r'^\d+$', merchant_name.strip()):
            return True
            
        # Check for common page identifiers
        page_indicators = ["page", "pg ", "p. ", "statement page"]
        merchant_lower = merchant_name.lower()
        for indicator in page_indicators:
            if indicator in merchant_lower:
                return True
                
        return False
    
    def categorize(self, merchant_name: str) -> Dict[str, any]:
        """Categorize a merchant name using BERT model
        
        Args:
            merchant_name: Raw merchant name
            
        Returns:
            Dictionary with category and confidence
        """
        try:
            # Check if this is an incoming payment (we only care about spending)
            if self._is_incoming_payment(merchant_name):
                logger.info(f"Skipping incoming payment: {merchant_name}")
                return {
                    "category": "others",
                    "confidence": 1.0,
                    "method": "filtered_incoming"
                }
                
            # Check if this is a page number or noise
            if self._is_page_number_or_noise(merchant_name):
                logger.info(f"Skipping page number or noise: {merchant_name}")
                return {
                    "category": "others",
                    "confidence": 1.0,
                    "method": "filtered_noise"
                }
            
            # Clean the merchant name
            cleaned_name = self._clean_merchant_name(merchant_name)
            logger.info(f"Cleaned merchant name: {merchant_name} -> {cleaned_name}")

            # Check if merchant name can be categorized using hardcoded rules
            hardcoded_category = self._categorize_brand_hardcoded(cleaned_name)
            if hardcoded_category:
                return {
                    "category": hardcoded_category,
                    "confidence": 1.0,
                    "method": "hardcoded"
                }
            
            # Use BERT model for prediction
            inputs = self.tokenizer(cleaned_name, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_category_idx = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_category_idx].item()
            
            # If confidence is below threshold, return "others" category
            if confidence < self.confidence_threshold:
                return {
                    "category": "others",
                    "confidence": confidence,
                    "method": "unconfident"
                }
            
            return {
                "category": self.categories[predicted_category_idx],
                "confidence": confidence,
                "method": "bert"
            }
        except Exception as e:
            logger.error(f"Error in BERT prediction: {str(e)}")
            raise
    
    def get_categories(self) -> List[str]:
        """Get all available spending categories
        
        Returns:
            List of all possible spending categories
        """
        return self.categories


# Example usage
if __name__ == "__main__":
    # Simple usage example for merchant categorization
    model_path = "models/merchant_categorizer"
    
    # Check if BERT model exists
    if not os.path.exists(model_path):
        logger.error(f"BERT model not found at {model_path}. Please train the model first using merchant_categorizer_trainer.py")
        exit(1)
    
    # Initialize categorizer
    categorizer = MerchantCategorizer(model_path)
    
    # Example merchant names
    example_merchants = [
        "NTUC FAIRPRICE",
        "GRAB TRANSPORT",
        "NETFLIX SG",
        "COLD STORAGE",
        "STARBUCKS COFFEE",
        "AMAZON PRIME"
    ]
    
    # Categorize each merchant
    print("\nMerchant Categorization Results:")
    print("-" * 50)
    for merchant in example_merchants:
        result = categorizer.categorize(merchant)
        print(f"Merchant: {merchant}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Method: {result['method']}")
        print("-" * 50) 
