"""
Credit Card Rewards Optimizer: Singapore Edition
Main package initialization
"""

# Import from statement_processing module
from src.statement_processing import MerchantCategorizer, MerchantCategorizerTrainer, PDFStatementParser, MerchantDataset

# Import from agent module
from src.agent import CardOptimizerAgent

# Import from model_context_protocol module
from src.model_context_protocol import CardOptimizerClient

__all__ = [
    # Statement processing
    'MerchantCategorizer',
    'MerchantCategorizerTrainer',
    'MerchantDataset',
    'PDFStatementParser',
    
    # Agent
    'CardOptimizerAgent',
    
    # Model Context Protocol
    'CardOptimizerClient',
] 