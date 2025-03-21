"""
Module for providing structured T&C information to other components.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from .tc_extractor import CardSummary, TCExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCProvider:
    """
    Class for providing structured T&C information to the synergy engine
    and other components of the application.
    """
    
    def __init__(self, summaries_dir: str = "./data/card_tcs/json"):
        """
        Initialize the TCProvider.
        
        Args:
            summaries_dir: Directory containing JSON summaries of T&C documents.
        """
        self.summaries_dir = summaries_dir
        self.summaries: Dict[str, CardSummary] = {}
        self._load_summaries()
    
    def _load_summaries(self) -> None:
        """Load all available T&C summaries."""
        extractor = TCExtractor(output_dir=self.summaries_dir)
        self.summaries = extractor.load_existing_summaries()
        logger.info(f"Loaded {len(self.summaries)} T&C summaries")
    
    def get_card_names(self) -> List[str]:
        """
        Get a list of available card names.
        
        Returns:
            List of card names.
        """
        return list(self.summaries.keys())
    
    def get_card_summary(self, card_name: str) -> Optional[CardSummary]:
        """
        Get the summary for a specific card.
        
        Args:
            card_name: Name of the card to retrieve.
            
        Returns:
            CardSummary object if found, None otherwise.
        """
        # Try exact match first
        if card_name in self.summaries:
            return self.summaries[card_name]
        
        # Try case-insensitive match
        for name, summary in self.summaries.items():
            if name.lower() == card_name.lower():
                return summary
        
        # Try partial match as a last resort
        for name, summary in self.summaries.items():
            if card_name.lower() in name.lower() or name.lower() in card_name.lower():
                return summary
        
        return None
    
    def get_card_fees(self, card_name: str) -> Optional[Dict[str, Any]]:
        """
        Get fee information for a specific card.
        
        Args:
            card_name: Name of the card.
            
        Returns:
            Dictionary of fee information if found, None otherwise.
        """
        summary = self.get_card_summary(card_name)
        if summary:
            return summary.fees.dict()
        return None
    
    def get_card_rewards(self, card_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get reward rates for a specific card.
        
        Args:
            card_name: Name of the card.
            
        Returns:
            List of reward rates if found, None otherwise.
        """
        summary = self.get_card_summary(card_name)
        if summary:
            return [rate.dict() for rate in summary.reward_rates]
        return None
    
    def get_card_benefits(self, card_name: str) -> Optional[Dict[str, Any]]:
        """
        Get benefits for a specific card.
        
        Args:
            card_name: Name of the card.
            
        Returns:
            Dictionary of benefits if found, None otherwise.
        """
        summary = self.get_card_summary(card_name)
        if summary:
            return summary.benefits.dict()
        return None
    
    def get_card_disclaimers(self, card_name: str) -> Optional[Dict[str, Any]]:
        """
        Get disclaimers for a specific card.
        
        Args:
            card_name: Name of the card.
            
        Returns:
            Dictionary of disclaimers if found, None otherwise.
        """
        summary = self.get_card_summary(card_name)
        if summary:
            return summary.disclaimers.dict()
        return None
    
    def get_reward_categories(self) -> List[str]:
        """
        Get a list of all unique reward categories across all cards.
        
        Returns:
            List of unique category names.
        """
        categories = set()
        for summary in self.summaries.values():
            for rate in summary.reward_rates:
                categories.add(rate.category)
        return sorted(list(categories))
    
    def get_cards_by_type(self, card_type: str) -> List[str]:
        """
        Get cards of a specific type.
        
        Args:
            card_type: Type of card (e.g., 'Cashback', 'Miles', 'Points').
            
        Returns:
            List of card names matching the type.
        """
        return [
            name for name, summary in self.summaries.items()
            if summary.card_type.lower() == card_type.lower()
        ]
    
    def get_cards_by_annual_fee(self, max_fee: float) -> List[str]:
        """
        Get cards with annual fee less than or equal to a specified amount.
        
        Args:
            max_fee: Maximum annual fee in SGD.
            
        Returns:
            List of card names with annual fee <= max_fee.
        """
        return [
            name for name, summary in self.summaries.items()
            if summary.fees.annual_fee <= max_fee
        ]
    
    def get_cards_by_minimum_income(self, min_income: float) -> List[str]:
        """
        Get cards with minimum income requirement <= specified amount.
        
        Args:
            min_income: Minimum income to check against.
            
        Returns:
            List of card names with minimum income <= min_income.
        """
        results = []
        for name, summary in self.summaries.items():
            min_income_str = summary.disclaimers.minimum_income
            try:
                # Extract numeric part and convert to float
                numeric_part = ''.join(c for c in min_income_str if c.isdigit() or c == '.')
                if numeric_part:
                    card_min_income = float(numeric_part)
                    if card_min_income <= min_income:
                        results.append(name)
            except ValueError:
                # If we can't parse the income, skip this card
                continue
                
        return results
    
    def refresh_summaries(self) -> None:
        """Reload summaries from disk."""
        self._load_summaries() 