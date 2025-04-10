from typing import List, Dict, Any, Optional, Union, BinaryIO
import os
import re
import logging
from datetime import datetime
from io import BytesIO

# PDF processing
import pdfplumber

# Local imports
from .merchant_categorizer import MerchantCategorizer

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
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file, is_path)
            
            # Parse transactions from text
            transactions = self.parse_text_with_llm(text)
            
            # Validate and clean transactions
            cleaned_transactions = self._clean_transactions(transactions)
            
            return cleaned_transactions
            
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
            Extracted text content as string
        """
        try:
            # Check if file exists when using path
            if is_path and not os.path.exists(pdf_file):
                logger.error(f"PDF file not found at {pdf_file}")
                return ""
                
            # Use pdfplumber to extract text
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
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
                # Validate required fields
                if not all(k in trans for k in ['merchant', 'amount', 'date', 'transaction_type']):
                    continue
                    
                # Clean merchant name
                merchant = trans['merchant'].strip()
                if not merchant:
                    continue
                    
                # Clean description - only handle simple text cleanup here
                description = self._clean_description(merchant)
                
                # Validate amount
                try:
                    amount = float(trans['amount'])
                    if amount <= 0:
                        continue
                except (ValueError, TypeError):
                    continue
                    
                # Validate and standardize date
                try:
                    date = self._standardize_date(trans['date'])
                    if not date:
                        continue
                except ValueError:
                    continue
                
                # Determine transaction type (withdrawal/deposit)
                trans_type = self._determine_transaction_type(trans['transaction_type'])
                
                # Use the merchant categorizer to categorize the merchant - delegate all categorization logic
                categorization = self.merchant_categorizer.categorize(description)
                
                cleaned.append({
                    'merchant': merchant,
                    'description': description,
                    'amount': amount,
                    'date': date,
                    'type': trans_type,
                    'category': categorization['category'],
                    'confidence': categorization['confidence'],
                    'method': categorization['method'],
                    'transaction_type': trans['transaction_type']
                })
            except Exception as e:
                logger.warning(f"Error cleaning transaction: {str(e)}")
                continue
                
        return cleaned
    
    def _standardize_date(self, date_str: str) -> Optional[str]:
        """Standardize date string to YYYY-MM-DD format
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Standardized date string or None if invalid
        """
        for pattern in self.date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    date_str = match.group(1)
                    # Handle different date formats
                    if '/' in date_str:
                        parts = date_str.split('/')
                    elif '-' in date_str:
                        parts = date_str.split('-')
                    else:
                        continue
                        
                    if len(parts) != 3:
                        continue
                        
                    # Convert to datetime object
                    if len(parts[2]) == 2:  # YY format
                        parts[2] = '20' + parts[2]
                        
                    date_obj = datetime.strptime(f"{parts[2]}-{parts[1]}-{parts[0]}", "%Y-%m-%d")
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        return None
    
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
    
    def parse_text_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Parse text into structured transaction data
        
        Args:
            text: Text content extracted from PDF
            
        Returns:
            List of structured transaction dictionaries
        """
        transactions = []
        lines = text.strip().split('\n')
        
        # Find transaction section
        transaction_section = False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for transaction section markers
            if any(marker in line.lower() for marker in ['transactions', 'transaction details', 'statement of account']):
                transaction_section = True
                i += 1
                continue
            
            if not transaction_section:
                i += 1
                continue
                
            # Skip header lines and empty lines
            if not line or any(header in line.lower() for header in ['date', 'description', 'amount', 'total']):
                i += 1
                continue
                
            # Try to parse transaction line
            try:
                # Look for date pattern
                date_match = None
                for pattern in self.date_patterns:
                    date_match = re.search(pattern, line)
                    if date_match:
                        break
                        
                if not date_match:
                    i += 1
                    continue
                    
                # Extract date
                date_str = date_match.group(1)
                
                # Remove date from line for further processing
                line = re.sub(pattern, '', line).strip()
                
                # Look for amount at the end of line
                amount_match = re.search(r'([\d,]+\.\d{2})', line)
                if not amount_match:
                    i += 1
                    continue
                    
                amount_str = amount_match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Extract transaction type (everything between date and amount)
                transaction_type = line[:line.rfind(amount_str)].strip()
                
                # Check if next line contains merchant name
                merchant = transaction_type
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Skip empty lines and lines with dates or amounts
                    if (next_line and 
                        not any(pattern in next_line for pattern in self.date_patterns) and
                        not re.search(r'([\d,]+\.\d{2})', next_line)):
                        merchant = next_line
                        i += 1  # Skip the merchant line in next iteration
                
                if merchant and amount > 0:
                        transactions.append({
                            'merchant': merchant,
                            'amount': amount,
                        'date': date_str,
                        'transaction_type': transaction_type
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing line: {line}, Error: {str(e)}")
            
            i += 1
                    
        return transactions


# Simple demo of the parser
if __name__ == "__main__":
    # Create an instance of the parser
    parser = PDFStatementParser()
    
    # Test with the sample PDF file
    pdf_path = "data/Sample Bank Statement.pdf"
    
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
