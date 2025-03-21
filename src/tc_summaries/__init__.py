"""
This module handles extraction of structured information from credit card T&C documents.
"""

from .tc_extractor import TCExtractor, CardSummary, CardFeeStructure, RewardRate, CardBenefits, ImportantDisclaimers
from .tc_provider import TCProvider 