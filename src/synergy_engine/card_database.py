from typing import List, Dict, Any, Optional
import os
import json
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We'll use a try/except here to maintain backwards compatibility
try:
    from src.tc_summaries import TCProvider
    TC_PROVIDER_AVAILABLE = True
except ImportError:
    logger.warning("TCProvider not available, falling back to legacy card data loading")
    TC_PROVIDER_AVAILABLE = False

class CardDatabase:
    """
    A class for managing the credit card database with reward rates and terms.
    """
    
    def __init__(self):
        """
        Initialize the CardDatabase.
        """
        self.cards = {}
        self.loaded = False
        self.tc_provider = None
        
        # Try to initialize the TCProvider if available
        if TC_PROVIDER_AVAILABLE:
            try:
                self.tc_provider = TCProvider()
                logger.info("TCProvider initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing TCProvider: {str(e)}")
    
    def load_cards(self, directory_path: Optional[str] = None) -> None:
        """
        Load credit card data from JSON files in a directory.
        
        Args:
            directory_path: Path to the directory containing card data files.
                           If None, defaults to data/card_tcs/json.
        """
        if directory_path is None:
            directory_path = os.path.join("data", "card_tcs", "json")
        
        # If TCProvider is available, try to use it first
        if self.tc_provider:
            try:
                self._load_cards_from_tc_provider()
                self.loaded = True
                return
            except Exception as e:
                logger.error(f"Error loading cards from TCProvider: {str(e)}")
                logger.info("Falling back to legacy card data loading")
        
        # Fallback to traditional JSON loading
        # Ensure directory exists
        os.makedirs(directory_path, exist_ok=True)
        
        # Find all JSON files in the directory
        file_paths = glob.glob(os.path.join(directory_path, "*.json"))
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    card_data = json.load(file)
                
                # Add card to the database
                card_id = card_data.get("id", os.path.basename(file_path).split('.')[0])
                self.cards[card_id] = card_data
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.loaded = True
    
    def _load_cards_from_tc_provider(self) -> None:
        """
        Load cards from the TCProvider if available.
        """
        if not self.tc_provider:
            raise ValueError("TCProvider not available")
        
        # Get all card names from the TC provider
        card_names = self.tc_provider.get_card_names()
        logger.info(f"Found {len(card_names)} cards in TCProvider")
        
        for card_name in card_names:
            try:
                # Get the full summary for the card
                summary = self.tc_provider.get_card_summary(card_name)
                if not summary:
                    logger.warning(f"No summary found for card: {card_name}")
                    continue
                
                # Convert to the format expected by the synergy engine
                card_id = card_name.lower().replace(" ", "_")
                
                # Extract reward type
                reward_type = summary.card_type.lower()
                
                # Build reward rates dictionary
                reward_rates = {}
                for rate in summary.reward_rates:
                    category = rate.category.lower()
                    # Try to extract numeric value from the rate string
                    rate_value = 0.0
                    rate_str = rate.rate
                    
                    # Handle different formats like "5% cashback" or "4 miles per $1"
                    if "%" in rate_str:
                        try:
                            rate_value = float(rate_str.split("%")[0].strip())
                        except ValueError:
                            pass
                    elif "miles" in rate_str or "points" in rate_str:
                        try:
                            parts = rate_str.split(" ")
                            for i, part in enumerate(parts):
                                if part.replace(".", "").isdigit():
                                    rate_value = float(part)
                                    break
                        except ValueError:
                            pass
                    
                    # Map category to standard categories used by the synergy engine
                    standard_category = self._map_to_standard_category(category)
                    if standard_category:
                        reward_rates[standard_category] = rate_value
                
                # Ensure general category exists
                if "general" not in reward_rates:
                    reward_rates["general"] = 1.0  # Default rate
                
                # Extract annual fee
                annual_fee = {
                    "amount": summary.fees.annual_fee,
                    "waiver_condition": summary.fees.annual_fee_waiver_condition
                }
                
                # Extract income requirement
                min_income_str = summary.disclaimers.minimum_income
                min_income = 0.0
                try:
                    # Extract numeric part and convert to float
                    numeric_part = ''.join(c for c in min_income_str if c.isdigit() or c == '.')
                    if numeric_part:
                        min_income = float(numeric_part)
                except ValueError:
                    pass
                
                # Build caps and limitations
                caps_and_limitations = {
                    "monthly_cap": 0.0,  # Default to no cap
                    "min_spend": 0.0,
                    "excluded_merchants": summary.disclaimers.excluded_transactions
                }
                
                # Try to extract min_spend from disclaimers
                min_spend_str = summary.disclaimers.minimum_spend or ""
                try:
                    # Extract numeric part and convert to float
                    numeric_part = ''.join(c for c in min_spend_str if c.isdigit() or c == '.')
                    if numeric_part:
                        caps_and_limitations["min_spend"] = float(numeric_part)
                except ValueError:
                    pass
                
                # Build sign-up bonus
                sign_up_bonus = {
                    "description": summary.benefits.welcome_bonus or "",
                    "value": 0.0,  # Hard to extract a numeric value here
                    "conditions": summary.benefits.welcome_bonus_conditions or ""
                }
                
                # Build final card data dictionary
                card_data = {
                    "id": card_id,
                    "name": summary.card_name,
                    "issuer": summary.issuer,
                    "reward_type": reward_type,
                    "annual_fee": annual_fee,
                    "requirements": {
                        "min_income": min_income,
                        "citizenship": []  # Not available in our schema
                    },
                    "reward_rates": reward_rates,
                    "caps_and_limitations": caps_and_limitations,
                    "sign_up_bonus": sign_up_bonus,
                    "additional_benefits": summary.benefits.other_benefits,
                    "terms_and_conditions_summary": "\n".join(summary.disclaimers.other_limitations)
                }
                
                # Add card to the database
                self.cards[card_id] = card_data
                logger.info(f"Loaded card from TC provider: {card_name}")
                
            except Exception as e:
                logger.error(f"Error processing card {card_name} from TC provider: {str(e)}")
    
    def _map_to_standard_category(self, category: str) -> Optional[str]:
        """
        Map a category from T&C to a standard category used by the synergy engine.
        
        Args:
            category: Category name from T&C.
            
        Returns:
            Mapped standard category or None if no mapping exists.
        """
        category = category.lower()
        
        # Define mapping dictionary
        category_mapping = {
            "dining": "dining",
            "restaurant": "dining",
            "food": "dining",
            "cafe": "dining",
            "groceries": "groceries",
            "supermarket": "groceries",
            "transport": "transport",
            "transportation": "transport",
            "commute": "transport",
            "grab": "transport",
            "taxi": "transport",
            "shopping": "shopping",
            "retail": "shopping",
            "online shopping": "shopping",
            "travel": "travel",
            "hotel": "travel",
            "airline": "travel",
            "flight": "travel",
            "bills": "bills",
            "utilities": "bills",
            "bill payment": "bills",
            "online": "online",
            "e-commerce": "online",
            "digital": "online",
            "overseas": "overseas",
            "foreign": "overseas",
            "international": "overseas",
            "general": "general",
            "all": "general",
            "other": "general"
        }
        
        # Try to find a match in the mapping
        for key, value in category_mapping.items():
            if key in category:
                return value
        
        # If no match, return None
        return None
    
    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a card by its ID.
        
        Args:
            card_id: ID of the card to retrieve.
        
        Returns:
            Card data dictionary or None if not found.
        """
        return self.cards.get(card_id)
    
    def save_card(self, card_data: Dict[str, Any], directory_path: Optional[str] = None) -> str:
        """
        Save a card to the database.
        
        Args:
            card_data: Card data dictionary.
            directory_path: Path to the directory to save the card data file.
                           If None, defaults to data/card_tcs/json.
        
        Returns:
            ID of the saved card.
        """
        if directory_path is None:
            directory_path = os.path.join("data", "card_tcs", "json")
        
        # Ensure directory exists
        os.makedirs(directory_path, exist_ok=True)
        
        # Get or generate card ID
        card_id = card_data.get("id", card_data.get("name", "").lower().replace(" ", "_"))
        card_data["id"] = card_id
        
        # Save card to database
        self.cards[card_id] = card_data
        
        # Save card to file
        file_path = os.path.join(directory_path, f"{card_id}.json")
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(card_data, file, indent=2)
        
        return card_id
    
    def get_all_cards(self) -> List[Dict[str, Any]]:
        """
        Get all cards in the database.
        
        Returns:
            List of all card data dictionaries.
        """
        return list(self.cards.values())
    
    def filter_cards(self, 
                     reward_type: Optional[str] = None, 
                     annual_fee_max: Optional[float] = None,
                     income_min: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Filter cards based on criteria.
        
        Args:
            reward_type: Type of reward to filter by (e.g., "miles", "cashback", "points").
            annual_fee_max: Maximum annual fee.
            income_min: Minimum income requirement.
        
        Returns:
            Filtered list of card data dictionaries.
        """
        filtered_cards = []
        
        for card in self.cards.values():
            # Filter by reward type
            if reward_type and reward_type.lower() not in card.get("reward_type", "").lower():
                continue
            
            # Filter by annual fee
            if annual_fee_max is not None:
                fee = card.get("annual_fee", {}).get("amount", 0)
                if fee > annual_fee_max:
                    continue
            
            # Filter by income requirement
            if income_min is not None:
                required_income = card.get("requirements", {}).get("min_income", 0)
                if required_income > income_min:
                    continue
            
            filtered_cards.append(card)
        
        return filtered_cards
    
    def get_card_template(self) -> Dict[str, Any]:
        """
        Get a template for a card data dictionary.
        
        Returns:
            Template card data dictionary.
        """
        return {
            "id": "",
            "name": "",
            "issuer": "",
            "reward_type": "",  # miles, cashback, points
            "annual_fee": {
                "amount": 0.0,
                "waiver_condition": ""
            },
            "requirements": {
                "min_income": 0.0,
                "citizenship": []
            },
            "reward_rates": {
                "dining": 0.0,
                "groceries": 0.0,
                "transport": 0.0,
                "shopping": 0.0,
                "travel": 0.0,
                "bills": 0.0,
                "online": 0.0,
                "overseas": 0.0,
                "general": 0.0  # Default rate for uncategorized spending
            },
            "caps_and_limitations": {
                "monthly_cap": 0.0,  # 0 means no cap
                "min_spend": 0.0,
                "excluded_merchants": []
            },
            "sign_up_bonus": {
                "description": "",
                "value": 0.0,
                "conditions": ""
            },
            "additional_benefits": [],
            "terms_and_conditions_summary": ""
        } 