import os
import json
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd
import polars as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import logging
from dotenv import load_dotenv
import openai
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext,
    Document
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY in .env file.")

# Set OpenAI API key
openai.api_key = api_key

# Constants for LLM configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1
CHUNK_SIZE = 521
CHUNK_OVERLAP = 10

# Initialize LLM and embedding models
llm = OpenAI(api_key=api_key, model=LLM_MODEL, temperature=TEMPERATURE)
embed_model = OpenAIEmbedding(api_key=api_key, model_name=EMBEDDING_MODEL)

class MerchantDataset(Dataset):
    """Dataset class for merchant categorization"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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
        
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the merchant categorizer model
        
        Args:
            model_path: Path to the trained BERT model
        """
        self.categories = self.CATEGORIES.copy()  # Instance copy of categories
        self.confidence_threshold = 0.0  # Lowered confidence threshold for better categorization
        
        # Location indicators to remove from merchant names
        self.location_indicators = [
            # Common location abbreviations
            "si ng", "sgp", "sg", "spore",
            
            # Country and city names
            "singapore", "singapura",
            
            # Educational institutions
            "nus", "ntu", "smu", "sit", "sim", "poly", "university", "college",
            
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
        if model_path and os.path.exists(model_path):
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
    
    def process_transactions(self, transactions: List[Dict]) -> Dict[str, float]:
        """Process a list of transactions to create a categorized spending profile
        
        Args:
            transactions: List of transaction dictionaries with merchant, amount, etc.
            
        Returns:
            Dictionary mapping spending categories to total amounts
        """
        spending_profile = {}
        
        # Initialize all categories with zero
        for category in self.get_categories():
            spending_profile[category] = 0.0
        
        # Process each transaction
        for transaction in transactions:
            merchant_name = transaction.get('merchant', '')
            amount = float(transaction.get('amount', 0))
            
            # Skip if amount is 0 or negative (outgoing payments are negative in many systems)
            if amount <= 0:
                logger.info(f"Skipping zero or negative amount transaction: {merchant_name} ({amount})")
                continue
                
            # Skip if this is an incoming transaction
            if self._is_incoming_payment(merchant_name):
                logger.info(f"Skipping incoming payment transaction: {merchant_name}")
                continue
                
            # Skip if this is a page number or noise
            if self._is_page_number_or_noise(merchant_name):
                logger.info(f"Skipping page number or noise: {merchant_name}")
                continue
            
            # Categorize merchant
            category_info = self.categorize(merchant_name)
            logger.info(f"Categorized merchant {merchant_name} as {category_info['category']} with confidence {category_info['confidence']}")
            category = category_info['category']
            
            # Add to spending profile - only if not filtered
            if category_info.get('method') not in ['filtered_incoming', 'filtered_noise']:
                spending_profile[category] += amount
        
        return spending_profile
    
    @classmethod
    def evaluate_gpt4(cls, test_data_path: str):
        """Evaluate GPT-4's performance on the same test data
        
        Args:
            test_data_path: Path to the test data
        """
        # Define path for saved GPT-4 results
        gpt4_results_path = "data/categorization/test/gpt4_evaluation_results.json"
        
        # Try to load existing results first
        if os.path.exists(gpt4_results_path):
            logger.info("Loading existing GPT-4 evaluation results...")
            with open(gpt4_results_path, 'r') as f:
                return json.load(f)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Prepare predictions and true labels
        true_labels = []
        predicted_labels = []
        predictions_data = []  # Store full prediction data
        
        for item in tqdm(test_data, desc="Evaluating GPT-4"):
            merchant_name = item['merchant_name']
            true_category = item['category']
            
            # Create prompt for GPT-4
            prompt = f"""Categorize this merchant name into one of these categories: {', '.join(cls.CATEGORIES)}.
            Only respond with the category name, nothing else.
            
            Merchant name: {merchant_name}"""
            
            try:
                # Call GPT-4
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a merchant categorization assistant. Respond only with the category name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                predicted_category = response.choices[0].message.content.strip().lower()
                
                # Validate prediction
                if predicted_category not in cls.CATEGORIES:
                    predicted_category = "others"
                
                true_labels.append(true_category)
                predicted_labels.append(predicted_category)
                
                # Store prediction data
                predictions_data.append({
                    'merchant_name': merchant_name,
                    'true_category': true_category,
                    'predicted_category': predicted_category
                })
                
            except Exception as e:
                logger.error(f"Error in GPT-4 prediction: {str(e)}")
                true_labels.append(true_category)
                predicted_labels.append("others")
                predictions_data.append({
                    'merchant_name': merchant_name,
                    'true_category': true_category,
                    'predicted_category': "others"
                })
        
        # Calculate classification report
        report = classification_report(true_labels, predicted_labels, 
                                    target_names=cls.CATEGORIES,
                                    labels=cls.CATEGORIES,
                                    output_dict=True)
        
        # Prepare results dictionary
        results = {
            'classification_report': report,
            'predictions': predictions_data
        }
        
        # Save results
        os.makedirs(os.path.dirname(gpt4_results_path), exist_ok=True)
        with open(gpt4_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print classification report
        print("\nGPT-4 Classification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                 target_names=cls.CATEGORIES,
                                 labels=cls.CATEGORIES))
        
        return results

    @classmethod
    def evaluate_model(cls, model_path: str, test_data_path: str):
        """Evaluate the trained model on test data and compare with GPT-4
        
        Args:
            model_path: Path to the trained BERT model
            test_data_path: Path to the test data
        """
        print("\n=== Evaluating BERT Model ===")
        bert_metrics = cls._evaluate_bert(model_path, test_data_path)
        
        print("\n=== Evaluating GPT-4 ===")
        gpt4_metrics = cls.evaluate_gpt4(test_data_path)
        
        # Compare results
        print("\n=== Model Comparison ===")
        print("BERT Model:")
        print(f"Accuracy: {bert_metrics['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {bert_metrics['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {bert_metrics['classification_report']['weighted avg']['f1-score']:.3f}")
        
        print("\nGPT-4:")
        print(f"Accuracy: {gpt4_metrics['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {gpt4_metrics['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {gpt4_metrics['classification_report']['weighted avg']['f1-score']:.3f}")
        
        return {
            'bert_metrics': bert_metrics,
            'gpt4_metrics': gpt4_metrics
        }

    @classmethod
    def _evaluate_bert(cls, model_path: str, test_data_path: str):
        """Internal method to evaluate BERT model performance
        
        Args:
            model_path: Path to the trained BERT model
            test_data_path: Path to the test data
        """
        # Initialize model
        model = cls(model_path)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Prepare predictions and true labels
        true_labels = []
        predicted_labels = []
        confidence_scores = []
        
        for item in tqdm(test_data, desc="Evaluating model"):
            merchant_name = item['merchant_name']
            true_category = item['category']
            
            try:
                # Use BERT model for prediction without fallback
                inputs = model.tokenizer(merchant_name, return_tensors="pt", padding=True, truncation=True)
                outputs = model.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_category_idx = torch.argmax(predictions).item()
                confidence = predictions[0][predicted_category_idx].item()
                
                predicted_category = model.categories[predicted_category_idx]
                
                true_labels.append(true_category)
                predicted_labels.append(predicted_category)
                confidence_scores.append(confidence)
                
            except Exception as e:
                logger.error(f"Error in BERT prediction: {str(e)}")
                # Skip this item instead of using rule-based fallback
                continue
        
        # Get unique categories in the test data
        unique_categories = sorted(set(true_labels + predicted_labels))
        
        # Calculate and print classification metrics
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                 target_names=unique_categories,
                                 labels=unique_categories))
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_categories)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_categories,
                    yticklabels=unique_categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Calculate average confidence per category
        confidence_by_category = {}
        for true, conf in zip(true_labels, confidence_scores):
            if true not in confidence_by_category:
                confidence_by_category[true] = []
            confidence_by_category[true].append(conf)
        
        print("\nAverage Confidence by Category:")
        for category, confidences in confidence_by_category.items():
            avg_conf = np.mean(confidences)
            print(f"{category}: {avg_conf:.3f}")
        
        return {
            'classification_report': classification_report(true_labels, predicted_labels, 
                                                         target_names=unique_categories,
                                                         labels=unique_categories,
                                                         output_dict=True),
            'confusion_matrix': cm,
            'confidence_by_category': {cat: np.mean(confs) for cat, confs in confidence_by_category.items()}
        }

# Example usage
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/categorization/training", exist_ok=True)
    os.makedirs("data/categorization/test", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Define paths for data files
    training_data_path = "data/categorization/training/merchant_training_data.json"
    test_data_path = "data/categorization/test/merchant_test_data.json"
    model_path = "models/merchant_categorizer"
    
    # Generate test data if it doesn't exist
    if not os.path.exists(test_data_path):
        try:
            logger.info("Generating test data...")
            MerchantCategorizer.generate_training_data(
                "data/categorization/ACRA.csv",
                test_data_path,
                is_training=False
            )
        except Exception as e:
            logger.error(f"Error generating test data: {str(e)}")
            exit(1)
    else:
        logger.info(f"Using existing test data from {test_data_path}")
    
    # Check if BERT model exists
    if not os.path.exists(model_path):
        logger.error(f"BERT model not found at {model_path}. Please train the model first.")
        exit(1)
    else:
        logger.info(f"Using existing BERT model from {model_path}")
    
    # Evaluate the model
    try:
        logger.info("Evaluating models...")
        metrics = MerchantCategorizer.evaluate_model(
            model_path,
            test_data_path
        )
        
        # Print overall metrics
        print("\n=== Overall Model Performance ===")
        print("\nBERT Model:")
        print(f"Accuracy: {metrics['bert_metrics']['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {metrics['bert_metrics']['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {metrics['bert_metrics']['classification_report']['weighted avg']['f1-score']:.3f}")
        
        print("\nGPT-4o Model:")
        print(f"Accuracy: {metrics['gpt4_metrics']['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {metrics['gpt4_metrics']['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {metrics['gpt4_metrics']['classification_report']['weighted avg']['f1-score']:.3f}")
        
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}")
        exit(1) 
