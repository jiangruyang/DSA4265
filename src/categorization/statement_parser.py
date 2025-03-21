import os
import re
import io
import csv
import logging
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import pandas as pd
import pypdf
from pdfminer.high_level import extract_text

# Configure logging
logger = logging.getLogger(__name__)

class StatementParser:
    """
    A class for parsing credit card statements from PDF and CSV files.
    """
    
    def __init__(self):
        """
        Initialize the StatementParser.
        """
        self.recognized_formats = {
            "dbs": self._parse_dbs_statement,
            "ocbc": self._parse_ocbc_statement,
            "uob": self._parse_uob_statement,
            "generic": self._parse_generic_statement
        }
    
    def parse_statement(self, file_content: bytes, file_format: str) -> List[Dict[str, Any]]:
        """
        Parse a credit card statement file.
        
        Args:
            file_content: Binary content of the statement file.
            file_format: Format of the file ('pdf' or 'csv').
        
        Returns:
            List of transaction dictionaries.
        """
        if file_format.lower() == 'pdf':
            return self._parse_pdf(file_content)
        elif file_format.lower() == 'csv':
            return self._parse_csv(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _parse_pdf(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Parse a PDF statement.
        
        Args:
            file_content: Binary content of the PDF file.
        
        Returns:
            List of transaction dictionaries.
        """
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_content)
        
        # Log the extracted text for debugging
        logger.debug(f"Extracted text from PDF: {text[:500]}...")
        
        # Try direct transaction extraction first (more reliable for our generated PDFs)
        transactions = self._extract_transactions_directly(text)
        if transactions:
            logger.debug(f"Successfully extracted {len(transactions)} transactions directly")
            return transactions
        
        # If direct extraction fails, try bank-specific parsers
        # Try to identify the bank format
        bank_format = self._identify_bank_format(text)
        logger.debug(f"Identified bank format: {bank_format}")
        
        # Use the appropriate parser for the identified format
        if bank_format in self.recognized_formats:
            parser_func = self.recognized_formats[bank_format]
            transactions = parser_func(text)
        else:
            # Fall back to generic parser
            transactions = self._parse_generic_statement(text)
        
        # Log the extracted transactions for debugging
        logger.debug(f"Extracted {len(transactions)} transactions")
        if transactions:
            logger.debug(f"First transaction: {transactions[0]}")
        
        # Ensure all transactions have the required fields
        for transaction in transactions:
            if 'date' not in transaction:
                transaction['date'] = datetime.now().strftime('%d-%m-%Y')
            if 'description' not in transaction:
                transaction['description'] = 'Unknown'
            if 'amount' not in transaction:
                transaction['amount'] = 0.0
        
        return transactions
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_content: Binary content of the PDF file.
        
        Returns:
            Extracted text.
        """
        # Try pdfminer.six first
        try:
            return extract_text(io.BytesIO(file_content))
        except Exception as e:
            logger.error(f"Error with pdfminer: {str(e)}")
            # Fall back to pypdf
            text = ""
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_content))
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF: {str(e)}")
            
            return text
    
    def _parse_csv(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Parse a CSV statement.
        
        Args:
            file_content: Binary content of the CSV file.
        
        Returns:
            List of transaction dictionaries.
        """
        try:
            # Try to read with pandas
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Try to identify key columns
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            desc_col = next((col for col in df.columns if any(term in col.lower() 
                                                              for term in ['desc', 'narration', 'merchant'])), None)
            amount_col = next((col for col in df.columns if any(term in col.lower() 
                                                                for term in ['amount', 'debit', 'spend'])), None)
            
            if not all([date_col, desc_col, amount_col]):
                # Fall back to positional if we can't identify the columns
                date_col = df.columns[0]
                desc_col = df.columns[1]
                amount_col = df.columns[2]
            
            transactions = []
            for _, row in df.iterrows():
                try:
                    # Parse date
                    date_str = str(row[date_col])
                    date = self._parse_date(date_str)
                    
                    # Parse description
                    description = str(row[desc_col]).strip()
                    
                    # Parse amount (handle negative amounts as positive for spending)
                    amount_str = str(row[amount_col]).replace(',', '')
                    amount = float(amount_str)
                    if amount < 0:
                        amount = abs(amount)
                    
                    transactions.append({
                        'date': date,
                        'description': description,
                        'amount': amount
                    })
                except (ValueError, TypeError) as e:
                    # Skip invalid rows
                    continue
            
            return transactions
            
        except Exception as e:
            # If pandas fails, try CSV reader
            try:
                csv_reader = csv.reader(io.StringIO(file_content.decode('utf-8')))
                rows = list(csv_reader)
                
                if len(rows) < 2:  # Need at least header + one data row
                    raise ValueError("CSV file has insufficient data")
                
                # Assume first row is header
                header = [col.lower() for col in rows[0]]
                
                # Try to identify key columns
                date_idx = next((i for i, col in enumerate(header) if 'date' in col), 0)
                desc_idx = next((i for i, col in enumerate(header) if any(term in col 
                                                                         for term in ['desc', 'narration', 'merchant'])), 1)
                amount_idx = next((i for i, col in enumerate(header) if any(term in col 
                                                                          for term in ['amount', 'debit', 'spend'])), 2)
                
                transactions = []
                for row in rows[1:]:  # Skip header
                    if len(row) <= max(date_idx, desc_idx, amount_idx):
                        continue  # Skip rows with insufficient columns
                    
                    try:
                        # Parse date
                        date_str = row[date_idx]
                        date = self._parse_date(date_str)
                        
                        # Parse description
                        description = row[desc_idx].strip()
                        
                        # Parse amount
                        amount_str = row[amount_idx].replace(',', '')
                        amount = float(amount_str)
                        if amount < 0:
                            amount = abs(amount)
                        
                        transactions.append({
                            'date': date,
                            'description': description,
                            'amount': amount
                        })
                    except (ValueError, TypeError):
                        # Skip invalid rows
                        continue
                
                return transactions
                
            except Exception as csv_e:
                raise ValueError(f"Failed to parse CSV file: {str(e)}, {str(csv_e)}")
    
    def _identify_bank_format(self, text: str) -> str:
        """
        Identify the bank format of a statement.
        
        Args:
            text: Extracted text from the statement.
        
        Returns:
            Identified bank format.
        """
        text_lower = text.lower()
        
        if "dbs" in text_lower or "development bank of singapore" in text_lower:
            return "dbs"
        elif "ocbc" in text_lower or "oversea-chinese banking corporation" in text_lower:
            return "ocbc"
        elif "uob" in text_lower or "united overseas bank" in text_lower:
            return "uob"
        else:
            return "generic"
    
    def _parse_dbs_statement(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse a DBS statement.
        
        Args:
            text: Extracted text from the statement.
        
        Returns:
            List of transaction dictionaries.
        """
        transactions = []
        
        try:
            # First try to parse transactions using table extraction
            # Pattern for transaction table with date, description, amount format
            # Look for Date followed by Description/Merchant and Amount
            lines = text.split('\n')
            transaction_started = False
            header_found = False
            
            date_idx, desc_idx, amount_idx = -1, -1, -1
            
            # Find the header row to determine column positions
            for i, line in enumerate(lines):
                if "Date" in line and "Description" in line and "Amount" in line:
                    header_found = True
                    header_parts = line.split()
                    for j, part in enumerate(header_parts):
                        if "Date" in part:
                            date_idx = j
                        if "Description" in part:
                            desc_idx = j
                        if "Amount" in part:
                            amount_idx = j
                    transaction_started = True
                    break
            
            if not header_found:
                # Try a different approach - look for date patterns
                date_pattern = r'\d{2}[-/]\d{2}[-/]\d{2,4}|\d{2}\s+[A-Za-z]{3}\s+\d{2,4}'
                amount_pattern = r'\$\s*\d+\.\d{2}|\d+\.\d{2}'
                
                for line in lines:
                    if re.search(date_pattern, line) and re.search(amount_pattern, line):
                        # This line likely contains a transaction
                        try:
                            # Extract date
                            date_match = re.search(date_pattern, line)
                            if date_match:
                                date_str = date_match.group(0)
                                
                            # Extract amount
                            amount_match = re.search(amount_pattern, line)
                            if amount_match:
                                amount_str = amount_match.group(0).replace('$', '').strip()
                                amount = float(amount_str)
                            
                            # Extract description (everything between date and amount)
                            start_idx = line.find(date_match.group(0)) + len(date_match.group(0))
                            end_idx = line.find(amount_match.group(0))
                            
                            if end_idx > start_idx:
                                description = line[start_idx:end_idx].strip()
                            else:
                                # Fall back to using everything after the date
                                description = line[start_idx:].replace(amount_match.group(0), '').strip()
                            
                            transactions.append({
                                'date': self._parse_date(date_str),
                                'description': description,
                                'amount': amount
                            })
                        except Exception as e:
                            logger.error(f"Error parsing transaction line: {line}, error: {str(e)}")
                            continue
            
            # Log if we found transactions
            if transactions:
                logger.debug(f"Found {len(transactions)} transactions with regex pattern matching approach")
                return transactions
            
            # If still no transactions, try our custom parser for generated PDFs
            return self._parse_generated_pdf_statement(text)
            
        except Exception as e:
            logger.error(f"Error in DBS statement parsing: {str(e)}")
            # Try generic parser as fallback
            return self._parse_generic_statement(text)
    
    def _parse_generated_pdf_statement(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse statements generated by our PDF generator.
        
        Args:
            text: Extracted text from the statement.
        
        Returns:
            List of transaction dictionaries.
        """
        transactions = []
        lines = text.split('\n')
        
        # Log the full content for debugging
        logger.debug(f"Attempting to parse generated PDF with {len(lines)} lines")
        logger.debug(f"Content preview: {text[:200]}")
        
        # Find the transaction table
        table_start_idx = -1
        table_end_idx = -1
        
        # First look for the table header
        for i, line in enumerate(lines):
            if "Date" in line and "Description" in line and "Amount" in line:
                table_start_idx = i
                logger.debug(f"Found table header at line {i}: {line}")
                break
        
        if table_start_idx > -1:
            # Find where the table ends (usually at "Total" row)
            for i in range(table_start_idx + 1, len(lines)):
                if "Total" in lines[i]:
                    table_end_idx = i
                    logger.debug(f"Found table end at line {i}: {lines[i]}")
                    break
            
            # If we couldn't find the end, use all remaining lines
            if table_end_idx == -1:
                table_end_idx = len(lines)
            
            # Process transaction lines
            for i in range(table_start_idx + 1, table_end_idx):
                line = lines[i].strip()
                if not line:
                    continue
                
                # Try to identify if this line looks like a transaction
                # Our format is typically:
                # 04-01-2025    IKEA TAMPINES    $250.00
                
                # Try to parse using regex patterns for date and amount
                date_pattern = r'\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{2}\s+[A-Za-z]{3}\s+\d{4}'
                amount_pattern = r'\$\s*\d+\.\d{2}|\d+\.\d{2}'
                
                date_match = re.search(date_pattern, line)
                amount_match = re.search(amount_pattern, line)
                
                if date_match and amount_match:
                    try:
                        # Extract date
                        date_str = date_match.group(0)
                        
                        # Extract amount - remove $ and commas
                        amount_str = amount_match.group(0).replace('$', '').replace(',', '')
                        amount = float(amount_str)
                        
                        # Extract description - everything between date and amount
                        date_end = line.find(date_match.group(0)) + len(date_match.group(0))
                        amount_start = line.find(amount_match.group(0))
                        
                        # If amount comes after date, extract description between them
                        if amount_start > date_end:
                            description = line[date_end:amount_start].strip()
                        else:
                            # Otherwise, try to extract description after date
                            parts = line.split()
                            if len(parts) >= 3:
                                # Skip the date part and the amount part, everything in between is description
                                description = ' '.join(parts[1:-1])
                            else:
                                description = "Unknown"
                        
                        # Log successful extraction
                        logger.debug(f"Extracted transaction: date={date_str}, desc={description}, amount={amount}")
                        
                        transactions.append({
                            'date': self._parse_date(date_str),
                            'description': description,
                            'amount': amount
                        })
                    except Exception as e:
                        logger.error(f"Error parsing line {line}: {str(e)}")
                        continue
        
        # If no transactions found with date/amount patterns, try a simpler approach
        if not transactions:
            logger.debug("No transactions found with pattern matching, trying simpler approach")
            # Simpler approach: Look for any line with a date-like string at the beginning
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:  # Need at least date, some description, and amount
                    first_part = parts[0]
                    # Check if the first part looks like a date (DD-MM-YYYY format)
                    if re.match(r'\d{2}-\d{2}-\d{4}', first_part):
                        try:
                            date_str = first_part
                            # Last part should be the amount
                            amount_str = parts[-1].replace('$', '').replace(',', '')
                            # Try to convert to float
                            amount = float(re.search(r'\d+\.\d{2}', amount_str).group(0) if re.search(r'\d+\.\d{2}', amount_str) else 0)
                            # Everything in between is the description
                            description = ' '.join(parts[1:-1])
                            
                            logger.debug(f"Extracted with simple approach: date={date_str}, desc={description}, amount={amount}")
                            
                            transactions.append({
                                'date': self._parse_date(date_str),
                                'description': description,
                                'amount': amount
                            })
                        except Exception as e:
                            logger.error(f"Error with simple approach on line {line}: {str(e)}")
                            continue
        
        logger.debug(f"Found {len(transactions)} transactions with generated PDF parser")
        return transactions
    
    def _parse_ocbc_statement(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse an OCBC statement.
        
        Args:
            text: Extracted text from the statement.
        
        Returns:
            List of transaction dictionaries.
        """
        # Try using the DBS parser first
        transactions = self._parse_dbs_statement(text)
        
        # If we got transactions, return them
        if transactions:
            return transactions
        
        # Otherwise, fall back to generic parser
        return self._parse_generic_statement(text)
    
    def _parse_uob_statement(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse a UOB statement.
        
        Args:
            text: Extracted text from the statement.
        
        Returns:
            List of transaction dictionaries.
        """
        # Try using the DBS parser first
        transactions = self._parse_dbs_statement(text)
        
        # If we got transactions, return them
        if transactions:
            return transactions
        
        # Otherwise, fall back to generic parser
        return self._parse_generic_statement(text)
    
    def _parse_generic_statement(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse a generic credit card statement.
        
        Args:
            text: Extracted text from the statement.
        
        Returns:
            List of transaction dictionaries.
        """
        # First try our generated PDF parser
        transactions = self._parse_generated_pdf_statement(text)
        
        if transactions:
            return transactions
        
        # If that fails, try to extract using regex patterns
        date_pattern = r'\d{2}[-/]\d{2}[-/]\d{2,4}|\d{2}\s+[A-Za-z]{3}\s+\d{2,4}'
        amount_pattern = r'\$\s*\d+\.\d{2}|\d+\.\d{2}'
        
        lines = text.split('\n')
        for line in lines:
            if re.search(date_pattern, line) and re.search(amount_pattern, line):
                # This line likely contains a transaction
                try:
                    # Extract date
                    date_match = re.search(date_pattern, line)
                    if date_match:
                        date_str = date_match.group(0)
                        
                    # Extract amount
                    amount_match = re.search(amount_pattern, line)
                    if amount_match:
                        amount_str = amount_match.group(0).replace('$', '').strip()
                        amount = float(amount_str)
                    
                    # Extract description (everything between date and amount)
                    start_idx = line.find(date_match.group(0)) + len(date_match.group(0))
                    end_idx = line.find(amount_match.group(0))
                    
                    if end_idx > start_idx:
                        description = line[start_idx:end_idx].strip()
                    else:
                        # Fall back to using everything after the date
                        description = line[start_idx:].replace(amount_match.group(0), '').strip()
                    
                    transactions.append({
                        'date': self._parse_date(date_str),
                        'description': description,
                        'amount': amount
                    })
                except Exception as e:
                    logger.error(f"Error parsing transaction line: {line}, error: {str(e)}")
                    continue
        
        return transactions
    
    def _parse_date(self, date_str: str) -> str:
        """
        Parse a date string into a standard format.
        
        Args:
            date_str: Date string to parse.
        
        Returns:
            Standardized date string.
        """
        try:
            # Try different date formats
            date_formats = [
                '%d-%m-%Y',  # DD-MM-YYYY
                '%d/%m/%Y',  # DD/MM/YYYY
                '%d-%m-%y',  # DD-MM-YY
                '%d/%m/%y',  # DD/MM/YY
                '%d %b %Y',  # DD MMM YYYY
                '%d %b %y',  # DD MMM YY
                '%d %B %Y',  # DD Month YYYY
                '%d %B %y'   # DD Month YY
            ]
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%d-%m-%Y')  # Return in standard format
                except ValueError:
                    continue
            
            # If none of the formats match, try extracting numbers
            date_parts = re.findall(r'\d+', date_str)
            if len(date_parts) >= 3:
                day = int(date_parts[0])
                month = int(date_parts[1])
                year = int(date_parts[2])
                if year < 100:
                    year += 2000
                return f"{day:02d}-{month:02d}-{year}"
            
            raise ValueError(f"Could not parse date: {date_str}")
        
        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {str(e)}")
            return datetime.now().strftime('%d-%m-%Y')  # Return today's date as fallback 

    def _extract_transactions_directly(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract transactions directly from text using pattern matching.
        This is more reliable for our generated PDF format.
        
        Args:
            text: Text extracted from PDF.
            
        Returns:
            List of transaction dictionaries.
        """
        transactions = []
        lines = text.split('\n')
        
        # Patterns for date and amount
        date_pattern = r'\d{2}-\d{2}-\d{4}'
        amount_pattern = r'\$\d+\.\d{2}'
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line has a date and amount
            date_match = re.search(date_pattern, line)
            amount_match = re.search(amount_pattern, line)
            
            if date_match and amount_match:
                try:
                    date_str = date_match.group(0)
                    amount_str = amount_match.group(0).replace('$', '')
                    amount = float(amount_str)
                    
                    # Extract description (between date and amount)
                    date_end_idx = line.find(date_match.group(0)) + len(date_match.group(0))
                    amount_start_idx = line.find(amount_match.group(0))
                    
                    if amount_start_idx > date_end_idx:
                        description = line[date_end_idx:amount_start_idx].strip()
                    else:
                        # If can't extract between, use parts
                        parts = line.split()
                        # Skip first part (date) and last part (amount), get everything in between
                        if len(parts) > 2:
                            description = ' '.join(parts[1:-1])
                        else:
                            description = "Unknown"
                    
                    logger.debug(f"Directly extracted: date={date_str}, desc={description}, amount={amount}")
                    
                    transactions.append({
                        'date': self._parse_date(date_str),
                        'description': description,
                        'amount': amount
                    })
                except Exception as e:
                    logger.error(f"Error extracting from line {line}: {str(e)}")
        
        return transactions 