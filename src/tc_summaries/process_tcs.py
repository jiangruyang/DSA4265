#!/usr/bin/env python3
"""
Script to process credit card T&C documents and extract structured information.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_pipeline.document_loader import DocumentLoader
from src.tc_summaries.tc_extractor import TCExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the T&C extraction process."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process credit card T&C documents.')
    parser.add_argument('--tc_dir', type=str, default='./data/card_tcs/pdf',
                        help='Directory containing T&C documents')
    parser.add_argument('--output_dir', type=str, default='./data/card_tcs/json',
                        help='Directory to save extracted JSON summaries')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reprocessing of all documents')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='LLM model to use for extraction')
    
    args = parser.parse_args()
    
    # Check if T&C directory exists
    if not os.path.exists(args.tc_dir):
        logger.error(f"T&C directory not found: {args.tc_dir}")
        return
    
    try:
        # Load the T&C documents
        logger.info(f"Loading T&C documents from {args.tc_dir}")
        document_loader = DocumentLoader()
        documents = document_loader.load_tc_documents(args.tc_dir)
        
        if not documents:
            logger.error("No T&C documents found.")
            return
        
        logger.info(f"Loaded {len(documents)} T&C documents")
        
        # Create the extractor
        tc_extractor = TCExtractor(
            model_name=args.model,
            output_dir=args.output_dir
        )
        
        # Process existing summaries
        existing_summaries = tc_extractor.load_existing_summaries()
        logger.info(f"Found {len(existing_summaries)} existing summaries")
        
        # Filter documents if not forcing reload
        if not args.force_reload and existing_summaries:
            filtered_documents = []
            for doc in documents:
                card_name = doc.metadata.get('card_name', '')
                if card_name not in existing_summaries:
                    filtered_documents.append(doc)
                    
            logger.info(f"{len(filtered_documents)} new documents to process")
            documents = filtered_documents
        
        if not documents:
            logger.info("No new documents to process")
            return
        
        # Process the documents
        logger.info("Extracting structured information from T&C documents...")
        summaries = tc_extractor.process_documents(documents)
        
        logger.info(f"Successfully processed {len(summaries)} documents")
        for card_name in summaries:
            logger.info(f"Processed: {card_name}")
            
    except Exception as e:
        logger.exception(f"Error processing T&C documents: {str(e)}")

if __name__ == "__main__":
    main() 