from typing import List, Dict, Any, Optional, Union, BinaryIO
import os
import re
import logging
from datetime import datetime
from io import BytesIO
import sys

# PDF processing
from monopoly.banks import BankDetector, banks
from monopoly.pdf import PdfDocument, PdfParser
from monopoly.pipeline import Pipeline
from monopoly.generic import GenericBank
import pandas as pd

# Add project root to Python path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Local imports
from src.statement_processing.merchant_categorizer import MerchantCategorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFStatementParser:
    """Parser for credit card statement PDFs
    
    This class provides functionality to extract transaction data from
    credit card statement PDFs. It converts PDFs to text and then
    uses pattern matching to extract structured transaction data.
    """
    
    def __init__(self, llm_engine: Optional[str] = None):
        """Initialize the PDF statement parser
        
        Args:
            llm_engine: Name of the LLM engine to use for parsing,
                        if None uses pattern matching implementation
        """
        self.llm_engine = llm_engine
        # Initialize merchant categorizer with the trained model
        model_path = "models/merchant_categorizer"
        self.merchant_categorizer = MerchantCategorizer(model_path=model_path)
        
        # Common date formats in statements
        self.date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})',  # DD/MM/YY or DD/MM/YYYY
            r'(\d{1,2}-\d{1,2}-\d{2,4})',  # DD-MM-YY or DD-MM-YYYY
            r'(\d{4}-\d{2}-\d{2})'          # YYYY-MM-DD
        ]
        
        # Transaction type patterns
        self.transaction_types = {
            'withdrawal': [
                'debit card transaction',
                'funds transfer',
                'fast payment',
                'atm withdrawal',
                'bill payment',
                'giro payment'
            ],
            'deposit': [
                'interest earned',
                'salary',
                'refund',
                'credit',
                'deposit'
            ]
        }
    
    def parse_statement(self, pdf_file: Union[str, BytesIO, BinaryIO], is_path: bool = True) -> List[Dict[str, Any]]:
        """Parse a credit card statement PDF into structured transaction data
        
        Args:
            pdf_file: Path to the PDF file or a file-like object containing PDF data
            is_path: Whether pdf_file is a file path (True) or file-like object (False)
            
        Returns:
            List of transaction dictionaries with fields:
            - merchant: Merchant name
            - amount: Transaction amount
            - date: Transaction date
            - type: Transaction type (withdrawal/deposit)
            - category: Merchant category
            - description: Cleaned transaction description
        """
        try:
            # Extract transactions from PDF
            df = self.extract_text_from_pdf(pdf_file, is_path)
            out = [
                {
                    'merchant': row['description'],
                    'amount': row['amount'],
                    'date': row['date'],
                    'type': 'withdrawal' if row['amount'] < 0 else 'deposit',
                    'category': 'TODO',
                    'description': row['description'],
                } for _, row in df.iterrows()
            ]
            print(out)
            out = self._clean_transactions(out)
            print(out)
            pd.DataFrame(out).to_csv('tmp/transactions.csv', index=False)
            return out
            
        except Exception as e:
            if is_path:
                error_context = f"Error parsing PDF statement at {pdf_file}: {str(e)}"
            else:
                error_context = f"Error parsing PDF statement from memory: {str(e)}"
            logger.error(error_context)
            return []
    
    def extract_text_from_pdf(self, pdf_file: Union[str, BytesIO, BinaryIO], is_path: bool = True) -> str:
        """Extract text content from a PDF file or file-like object
        
        Args:
            pdf_file: Path to the PDF file or a file-like object containing PDF data
            is_path: Whether pdf_file is a file path (True) or file-like object (False)
            
        Returns:
            pd.DataFrame: A pandas DataFrame containing the extracted transactions
        """
        try:
            # Check if file exists when using path
            if is_path and not os.path.exists(pdf_file):
                logger.error(f"PDF file not found at {pdf_file}")
                return ""

            with open("tmp/Sample Bank Statement.pdf", "rb") as f:
                pdf_bytes = BytesIO(f.read())
                
            document = PdfDocument(file_bytes=pdf_bytes)
            
            bank_class = BankDetector(document).detect_bank(banks)
            parser = PdfParser(bank_class, document)
            pipeline = Pipeline(parser)
            statement = pipeline.extract(safety_check=False)
            transactions = pipeline.transform(statement)
            
            return pd.DataFrame(transactions)

            
        except Exception as e:
            if is_path:
                logger.error(f"Error extracting text from PDF at {pdf_file}: {str(e)}")
            else:
                logger.error(f"Error extracting text from PDF in memory: {str(e)}")
            return ""
    
    def _determine_transaction_type(self, description: str) -> str:
        """Determine if a transaction is a withdrawal or deposit
        
        Args:
            description: Transaction description
            
        Returns:
            'withdrawal' or 'deposit'
        """
        description_lower = description.lower()
        
        # Check for deposit indicators
        for deposit_term in self.transaction_types['deposit']:
            if deposit_term in description_lower:
                return 'deposit'
                
        # Default to withdrawal
        return 'withdrawal'
    
    def _clean_description(self, description: str) -> str:
        """Clean up transaction description
        
        Args:
            description: Raw transaction description
            
        Returns:
            Cleaned description
        """
        # Remove common patterns
        patterns_to_remove = [
            r'\d{1,3}(,\d{3})*(\.\d{2})?\s+\d{1,3}(,\d{3})*(\.\d{2})?',  # Remove balance patterns
            r'/\s*Receipt\s*\d*',  # Remove receipt numbers
            r'\s*/\s*$',  # Remove trailing slashes
            r'\s+',  # Normalize whitespace
        ]
        
        cleaned = description
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, ' ', cleaned)
            
        return cleaned.strip()
    
    def _clean_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate transaction data
        
        Args:
            transactions: List of raw transaction dictionaries
            
        Returns:
            List of cleaned and validated transaction dictionaries
        """
        cleaned = []
        for trans in transactions:
            try:
                # Use the merchant categorizer to categorize the merchant - delegate all categorization logic
                categorization = self.merchant_categorizer.categorize(trans['description'])
                
                cleaned.append({
                    'merchant': trans['merchant'],
                    'description': trans['description'],
                    'amount': trans['amount'],
                    'date': trans['date'],
                    'type': trans['type'],
                    'category': categorization['category'],
                    'confidence': categorization['confidence'],
                    'method': categorization['method'],
                    'transaction_type': trans['type']
                })
            except Exception as e:
                logger.warning(f"Error cleaning transaction: {str(e)}")
                continue
                
        return cleaned

    
    def process_transactions(self, transactions: List[Dict]) -> Dict[str, float]:
        """Process a list of transactions to create a categorized spending profile
        
        Args:
            transactions: List of transaction dictionaries with merchant, amount, etc.
            
        Returns:
            Dictionary mapping spending categories to total amounts
        """
        spending_profile = {}
        
        # Initialize all categories with zero based on merchant categorizer categories
        for category in self.merchant_categorizer.get_categories():
            spending_profile[category] = 0.0
        
        # Process each transaction
        for transaction in transactions:
            merchant_name = transaction.get('merchant', '')
            amount = float(transaction.get('amount', 0))
            
            # Skip if amount is 0 or negative (outgoing payments are negative in many systems)
            if amount <= 0:
                logger.info(f"Skipping zero or negative amount transaction: {merchant_name} ({amount})")
                continue
                
            # Get category from the transaction if it exists, otherwise categorize
            if 'category' in transaction and transaction['category']:
                category = transaction['category']
                method = transaction.get('method', 'existing')
            else:
                # Merchant not yet categorized, do it now
                category_info = self.merchant_categorizer.categorize(merchant_name)
                category = category_info['category']
                method = category_info['method']
            
            logger.info(f"Transaction {merchant_name} categorized as {category} with method {method}")
            
            # Add to spending profile - only if not filtered
            if method not in ['filtered_incoming', 'filtered_noise']:
                spending_profile[category] += amount
        
        return spending_profile


# Simple demo of the parser
if __name__ == "__main__":
    print("Starting PDF statement parser")
    
    # Create an instance of the parser
    parser = PDFStatementParser()
    
    # Test with the sample PDF file
    pdf_path = "tmp/Sample Bank Statement.pdf"
    
    # Parse the statement
    print(f"Testing PDF parser with file: {pdf_path}")
    transactions = parser.parse_statement(pdf_path)
    
    # Display the extracted transactions with categorization
    print("\nExtracted Transactions:")
    print("\nWithdrawals:")
    for i, transaction in enumerate(transactions, 1):
        if transaction['type'] == 'withdrawal':
            print(f"{i}. {transaction['date']} - {transaction['transaction_type']}")
            print(f"   Merchant: {transaction['description']}")
            print(f"   Category: {transaction['category']}")
            print(f"   Amount: ${transaction['amount']:.2f}")
            print()
    
    print("\nDeposits:")
    for i, transaction in enumerate(transactions, 1):
        if transaction['type'] == 'deposit':
            print(f"{i}. {transaction['date']} - {transaction['transaction_type']}")
            print(f"   Merchant: {transaction['description']}")
            print(f"   Category: {transaction['category']}")
            print(f"   Amount: ${transaction['amount']:.2f}")
            print()
    
    # Print summary
    total_withdrawals = sum(t['amount'] for t in transactions if t['type'] == 'withdrawal')
    total_deposits = sum(t['amount'] for t in transactions if t['type'] == 'deposit')
    print(f"\nSummary:")
    print(f"Total Withdrawals: ${total_withdrawals:.2f}")
    print(f"Total Deposits: ${total_deposits:.2f}")
    print(f"Net Change: ${total_deposits - total_withdrawals:.2f}") 
