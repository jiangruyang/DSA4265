import os
from typing import List, Dict, Any
import pandas as pd
from langchain.schema import HumanMessage, SystemMessage
from src.chat_agent.llm_config import get_llm

class TransactionClassifier:
    """
    A class for classifying credit card transactions into spending categories using LLM.
    """
    
    def __init__(self, batch_size: int = 30):
        """
        Initialize the TransactionClassifier.
        
        Args:
            batch_size: Maximum number of transactions to process in a single LLM call.
        """
        self.batch_size = batch_size
        self.llm = get_llm(temperature=0.0)
        self.categories = [
            "Dining", "Groceries", "Transport", "Shopping", 
            "Travel", "Bills & Utilities", "Entertainment", "Others"
        ]
    
    def classify_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a list of transactions into spending categories.
        
        Args:
            transactions: List of transaction dictionaries containing at least 'description' and 'amount' keys.
        
        Returns:
            List of transaction dictionaries with added 'category' key.
        """
        # Process transactions in batches
        classified_transactions = []
        for i in range(0, len(transactions), self.batch_size):
            batch = transactions[i:i+self.batch_size]
            classified_batch = self._classify_batch(batch)
            classified_transactions.extend(classified_batch)
        
        return classified_transactions
    
    def _classify_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of transactions using the LLM.
        
        Args:
            transactions: Batch of transaction dictionaries.
        
        Returns:
            Classified transaction dictionaries.
        """
        # Prepare the prompt
        transaction_text = self._format_transactions(transactions)
        system_message = self._get_system_prompt()
        prompt = f"Please classify the following transactions into categories:\n\n{transaction_text}"
        
        # Get LLM response
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        
        # Parse the response
        return self._parse_classification_response(response.content, transactions)
    
    def _format_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """
        Format transactions for the LLM prompt.
        
        Args:
            transactions: List of transaction dictionaries.
        
        Returns:
            Formatted transaction text.
        """
        formatted_lines = []
        for i, t in enumerate(transactions):
            formatted_lines.append(f"{i+1}. {t['date']} - {t['description']} - ${t['amount']:.2f}")
        
        return "\n".join(formatted_lines)
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for transaction classification.
        
        Returns:
            System prompt string.
        """
        categories_str = ", ".join(self.categories)
        
        return f"""You are an expert financial transaction classifier for Singapore.
Your task is to categorize each transaction into one of the following categories:
{categories_str}

For each transaction, output the transaction number followed by the category in this exact format:
<TRANSACTION_NUM>: <CATEGORY>

For example:
1: Dining
2: Groceries
3: Others

Guidelines for classification:
- Dining: Restaurants, cafes, food delivery, bars
- Groceries: Supermarkets, grocery stores, convenience stores
- Transport: Public transit, taxis, ride-sharing, fuel
- Shopping: Retail stores, online shopping, clothing, electronics
- Travel: Flights, hotels, travel booking, overseas transactions
- Bills & Utilities: Phone bills, internet, utilities, insurance
- Entertainment: Movies, events, streaming services, games
- Others: Any transaction that doesn't clearly fit the above categories

Reply with ONLY the transaction number and category, nothing else."""
    
    def _parse_classification_response(self, response: str, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse the LLM classification response.
        
        Args:
            response: The text response from the LLM.
            transactions: The original transaction dictionaries.
        
        Returns:
            Updated transaction dictionaries with categories.
        """
        # Create a copy of the original transactions
        classified_transactions = transactions.copy()
        
        # Parse the response lines
        lines = response.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            # Try to extract transaction number and category
            try:
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                    
                transaction_num = int(parts[0].strip()) - 1  # Adjust for 0-based indexing
                category = parts[1].strip()
                
                if 0 <= transaction_num < len(classified_transactions):
                    classified_transactions[transaction_num]['category'] = category
                
            except ValueError:
                # Skip lines that don't match the expected format
                continue
        
        # Set default category for any transactions that weren't classified
        for transaction in classified_transactions:
            if 'category' not in transaction:
                transaction['category'] = 'Others'
        
        return classified_transactions
    
    def aggregate_by_category(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate transaction amounts by category.
        
        Args:
            transactions: List of classified transaction dictionaries.
        
        Returns:
            Dictionary mapping categories to total amounts.
        """
        category_totals = {category: 0.0 for category in self.categories}
        
        for transaction in transactions:
            category = transaction.get('category', 'Others')
            amount = transaction.get('amount', 0.0)
            
            if category in category_totals:
                category_totals[category] += amount
            else:
                category_totals['Others'] += amount
        
        return category_totals 