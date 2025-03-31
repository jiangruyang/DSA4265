import os
import json
from typing import Dict, List, Tuple, Optional

class MerchantCategorizer:
    """A distilled model for categorizing merchants into spending categories
    
    This class implements a lightweight model for merchant categorization
    using knowledge distilled from a larger teacher model trained on
    ACRA business data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the merchant categorizer model
        
        Args:
            model_path: Path to the model file, if None uses dummy implementation
        """
        self.model_path = model_path
        self.categories = [
            "groceries",
            "dining",
            "transportation",
            "shopping",
            "entertainment",
            "travel",
            "utilities",
            "healthcare",
            "education",
            "others"
        ]
        
        # Load merchant keyword mappings (dummy implementation)
        self.merchant_categories = {
            "fairprice": "groceries",
            "ntuc": "groceries",
            "cold storage": "groceries",
            "giant": "groceries",
            "sheng siong": "groceries",
            
            "grab": "transportation",
            "gojek": "transportation",
            "comfort": "transportation",
            "taxi": "transportation",
            "uber": "transportation",
            
            "mcdonald": "dining",
            "kfc": "dining",
            "starbucks": "dining",
            "restaurant": "dining",
            "cafe": "dining",
            "food": "dining",
            
            "uniqlo": "shopping",
            "zara": "shopping",
            "h&m": "shopping",
            "takashimaya": "shopping",
            "courts": "shopping",
            
            "netflix": "entertainment",
            "spotify": "entertainment",
            "disney": "entertainment",
            "cinema": "entertainment",
            "ticket": "entertainment"
        }
    
    def categorize(self, merchant_name: str) -> Dict[str, any]:
        """Categorize a merchant name into a spending category
        
        Args:
            merchant_name: The name of the merchant to categorize
            
        Returns:
            Dict containing the category and confidence score
        """
        # This is a dummy implementation - in a real implementation,
        # this would use the trained student model to infer the category
        merchant_name_lower = merchant_name.lower()
        
        # Simple keyword matching
        for keyword, category in self.merchant_categories.items():
            if keyword in merchant_name_lower:
                return {
                    "category": category,
                    "confidence": 0.9  # Fixed confidence for dummy implementation
                }
        
        # Default category if no match found
        return {
            "category": "others",
            "confidence": 0.5
        }
    
    def get_categories(self) -> List[str]:
        """Get all available spending categories
        
        Returns:
            List of all possible spending categories
        """
        return self.categories
    
    def process_transactions(self, transactions: List[Dict]) -> Dict[str, float]:
        """Process a list of transactions to create a categorized spending profile
        
        Args:
            transactions: List of transaction dictionaries with merchant, amount, etc.
            
        Returns:
            Dictionary mapping spending categories to total amounts
        """
        spending_profile = {}
        
        # Initialize all categories with zero
        for category in self.get_categories():
            spending_profile[category] = 0.0
        
        # Process each transaction
        for transaction in transactions:
            merchant_name = transaction.get('merchant', '')
            amount = float(transaction.get('amount', 0))
            
            # Categorize merchant
            category_info = self.categorize(merchant_name)
            category = category_info['category']
            
            # Add to spending profile
            spending_profile[category] += amount
        
        return spending_profile
    
    @classmethod
    def train(cls, 
              teacher_model_name: str, 
              acra_data_path: str, 
              output_model_path: str) -> 'MerchantCategorizer':
        """Train a student model using knowledge distillation from a teacher model
        
        Args:
            teacher_model_name: The name of the teacher model to use
            acra_data_path: Path to the ACRA business data
            output_model_path: Path to save the trained student model
            
        Returns:
            Trained MerchantCategorizer instance
        """
        # This is a placeholder for the actual training code
        # In a real implementation, this would:
        # 1. Load the teacher model
        # 2. Use it to generate labels for the ACRA data
        # 3. Train a smaller student model on these labels
        # 4. Save the student model to output_model_path
        
        print(f"Training student model using {teacher_model_name} on {acra_data_path}")
        print(f"Model will be saved to {output_model_path}")
        
        # Create a dummy model file
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        with open(output_model_path, 'w') as f:
            json.dump({"model_type": "merchant_categorizer", "version": "0.1"}, f)
        
        return cls(model_path=output_model_path)


# Simple demo of the model
if __name__ == "__main__":
    # Create an instance of the model
    model = MerchantCategorizer()
    
    # Test some merchant names
    test_merchants = [
        "NTUC FairPrice",
        "Grab Transport",
        "McDonald's",
        "Uniqlo Somerset",
        "Netflix Subscription",
        "Unknown Merchant"
    ]
    
    for merchant in test_merchants:
        result = model.categorize(merchant)
        print(f"{merchant}: {result['category']} (Confidence: {result['confidence']})")
    
    # Test transaction processing
    transactions = [
        {'merchant': 'NTUC FairPrice', 'amount': 200.50},
        {'merchant': 'Grab Transport', 'amount': 150.75},
        {'merchant': 'McDonald\'s', 'amount': 25.60},
        {'merchant': 'Uniqlo Somerset', 'amount': 120.00},
        {'merchant': 'Netflix Subscription', 'amount': 19.90},
    ]
    
    spending_profile = model.process_transactions(transactions)
    print("\nProcessed Spending Profile:")
    for category, amount in sorted(spending_profile.items(), key=lambda x: x[1], reverse=True):
        if amount > 0:
            print(f"- {category}: ${amount:.2f}")
    
    # Print available categories
    print("\nAvailable Categories:")
    print(", ".join(model.get_categories())) 