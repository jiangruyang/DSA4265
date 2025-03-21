#!/usr/bin/env python3
"""
Test script for the T&C extraction module.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_pipeline.document_loader import DocumentLoader
from src.tc_summaries.tc_extractor import TCExtractor
from src.tc_summaries.tc_provider import TCProvider
from src.synergy_engine.card_database import CardDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test the T&C extraction modules."""
    # Load environment variables
    load_dotenv()
    
    # 1. Test document loading
    logger.info("Testing document loading...")
    document_loader = DocumentLoader()
    tc_dir = "./data/card_tcs/pdf"
    
    if not os.path.exists(tc_dir) or len(os.listdir(tc_dir)) == 0:
        logger.error(f"No T&C documents found in {tc_dir}")
        return
    
    documents = document_loader.load_tc_documents(tc_dir)
    logger.info(f"Loaded {len(documents)} documents")
    
    # 2. Test TC extraction
    logger.info("Testing T&C extraction...")
    tc_extractor = TCExtractor()
    
    for doc in documents:
        try:
            logger.info(f"Extracting T&C summary for {doc.metadata.get('card_name', 'unknown card')}")
            summary = tc_extractor.extract_tc_summary(doc)
            
            # Save the summary
            file_path = tc_extractor.save_summary(summary)
            logger.info(f"Saved summary to {file_path}")
            
            # Print some key information
            logger.info(f"Card: {summary.card_name}")
            logger.info(f"  Issuer: {summary.issuer}")
            logger.info(f"  Type: {summary.card_type}")
            logger.info(f"  Annual Fee: ${summary.fees.annual_fee}")
            logger.info(f"  Number of reward categories: {len(summary.reward_rates)}")
            
        except Exception as e:
            logger.error(f"Error extracting T&C for {doc.metadata.get('source', 'unknown')}: {str(e)}")
    
    # 3. Test TCProvider
    logger.info("Testing TCProvider...")
    tc_provider = TCProvider()
    card_names = tc_provider.get_card_names()
    logger.info(f"TCProvider found {len(card_names)} cards")
    
    for card_name in card_names:
        try:
            logger.info(f"Getting details for {card_name}")
            summary = tc_provider.get_card_summary(card_name)
            
            if summary:
                logger.info(f"  Card Type: {summary.card_type}")
                logger.info(f"  Annual Fee: ${summary.fees.annual_fee}")
                logger.info(f"  Minimum Income: {summary.disclaimers.minimum_income}")
                
                # Also get some specific components
                fees = tc_provider.get_card_fees(card_name)
                rewards = tc_provider.get_card_rewards(card_name)
                
                logger.info(f"  Fees: {fees}")
                logger.info(f"  Number of reward rates: {len(rewards) if rewards else 0}")
        except Exception as e:
            logger.error(f"Error getting details for {card_name}: {str(e)}")
    
    # 4. Test integration with CardDatabase
    logger.info("Testing integration with CardDatabase...")
    card_db = CardDatabase()
    card_db.load_cards()
    
    all_cards = card_db.get_all_cards()
    logger.info(f"CardDatabase loaded {len(all_cards)} cards")
    
    # Print some information about the loaded cards
    for card in all_cards[:3]:  # Print details for up to 3 cards
        logger.info(f"Card: {card.get('name', 'Unknown')}")
        logger.info(f"  Reward Type: {card.get('reward_type', 'Unknown')}")
        logger.info(f"  Annual Fee: ${card.get('annual_fee', {}).get('amount', 'Unknown')}")
        logger.info(f"  Reward Rates: {card.get('reward_rates', {})}")
    
    # 5. Test filtering
    if all_cards:
        logger.info("Testing card filtering...")
        
        # Filter by reward type
        cashback_cards = card_db.filter_cards(reward_type="cashback")
        logger.info(f"Found {len(cashback_cards)} cashback cards")
        
        # Filter by annual fee
        low_fee_cards = card_db.filter_cards(annual_fee_max=100)
        logger.info(f"Found {len(low_fee_cards)} cards with annual fee <= $100")

if __name__ == "__main__":
    main() 