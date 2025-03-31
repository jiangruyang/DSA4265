from typing import Dict, List, Any, Optional
import asyncio
from src.mcp.client import CardOptimizerClient

class CardOptimizerAgent:
    """Main agent implementation for the Credit Card Rewards Optimizer
    
    This agent is responsible for recommending optimal card combinations
    and usage strategies based on the user's spending profile and preferences.
    It leverages the MCP tools for card data access and operates on pre-categorized
    spending data. The agent performs reasoning directly using an LLM instead of 
    relying on a separate synergy engine.
    """
    
    def __init__(self):
        """Initialize the agent with the client for tool access"""
        self.client = CardOptimizerClient()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent by ensuring client is ready"""
        if not self.initialized:
            await self.client.initialize()
            self.initialized = True
    
    async def recommend_cards(self, 
                        spending_profile: Dict[str, float], 
                        preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal card combinations based on spending and preferences
        
        Args:
            spending_profile: Dictionary mapping categories to spending amounts (preprocessed)
            preferences: User preferences including reward type, annual fee tolerance, etc.
            
        Returns:
            Recommendation dictionary with optimal cards and usage strategy
        """
        await self.initialize()
        
        # Get available cards
        available_cards = await self.client.get_available_cards()
        
        # Filter cards based on preferences
        filtered_cards = []
        for card in available_cards:
            # Filter by reward type if specified
            if 'type' in preferences and preferences['type'] != card['type']:
                continue
            
            # Filter by annual fee tolerance
            if 'max_annual_fee' in preferences:
                if card['annual_fee'] > preferences['max_annual_fee']:
                    continue
            
            # Apply income filtering if available
            if 'income' in preferences and preferences['income'] < card.get('min_income', 0):
                continue
            
            filtered_cards.append(card)
        
        # Get details for filtered cards
        card_details = []
        for card in filtered_cards:
            card_id = card['id']
            card_details.append(await self.client.get_card_details(card_id))
        
        # In a real implementation, this would use direct LLM reasoning through MCP
        # This is a dummy implementation to simulate the LLM-based reasoning
        
        # Dummy implementation: select cards and create a recommendation
        if len(card_details) >= 2:
            # For demonstration, we'll always recommend the first two cards
            card1 = card_details[0]
            card2 = card_details[1]
            
            # For miles cards, focus on travel categories
            if card1['type'] == 'miles':
                card1_categories = ["travel", "transportation"]
                card1_reasons = ["higher miles earning rate for travel", "good for transportation"]
            else:
                card1_categories = ["dining", "shopping"]
                card1_reasons = ["good cashback for dining", "rewards for shopping"] 
                
            # For the second card, prioritize different categories
            if card2['type'] == 'cashback':
                card2_categories = ["groceries", "dining"]
                card2_reasons = ["higher cashback for groceries", "good for dining expenses"]
            else:
                card2_categories = ["shopping", "entertainment"]
                card2_reasons = ["bonus points for shopping", "rewards for entertainment"]
            
            # Create usage strategy
            usage_strategy = {
                card1['id']: [
                    {"category": card1_categories[0], "reason": card1_reasons[0]},
                    {"category": card1_categories[1], "reason": card1_reasons[1]}
                ],
                card2['id']: [
                    {"category": card2_categories[0], "reason": card2_reasons[0]},
                    {"category": card2_categories[1], "reason": card2_reasons[1]}
                ]
            }
            
            # Calculate estimated monthly value (dummy implementation)
            # In reality, this would be calculated based on spending profile and card rewards
            estimated_value = sum(spending_profile.values()) * 0.03  # Assume 3% average rewards
            
            # Create final recommendation
            return {
                'top_cards': [
                    {"id": card1['id'], "name": card1['name'], "priority": 1},
                    {"id": card2['id'], "name": card2['name'], "priority": 2}
                ],
                'usage_strategy': usage_strategy,
                'estimated_value': estimated_value,
                'explanation': "This recommendation is based on your spending profile and preferences. " +
                              "In a production environment, this would be generated through LLM reasoning."
            }
        else:
            # If there are too few cards, return a simple recommendation
            return {
                'top_cards': [{"id": card_details[0]['id'], "name": card_details[0]['name'], "priority": 1}] if card_details else [],
                'usage_strategy': {},
                'estimated_value': 0.0,
                'explanation': "Not enough eligible cards found matching your preferences."
            }
    
    async def query_terms_and_conditions(self, question: str, card_id: str) -> Dict[str, Any]:
        """Query the terms and conditions for a specific card
        
        Args:
            question: Natural language question about the card
            card_id: ID of the card to query about
            
        Returns:
            Response dictionary with answer, source, and confidence
        """
        await self.initialize()
        return await self.client.query_tc(question, card_id)
    
    async def analyze_scenario(self, 
                        current_spending: Dict[str, float],
                        scenario_change: str,
                        current_recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how a spending scenario change would affect recommendations
        
        Args:
            current_spending: Current spending profile
            scenario_change: Natural language description of the spending change
            current_recommendation: Current card recommendation
            
        Returns:
            Updated recommendation based on the scenario
        """
        await self.initialize()
        
        # In a real implementation, this would use LLM reasoning through MCP
        # This is a dummy implementation to simulate the agent's reasoning
        
        # Simple parsing of scenario change (very limited)
        modified_spending = current_spending.copy()
        
        # Dummy logic to simulate scenario analysis
        if "double" in scenario_change.lower() and "dining" in scenario_change.lower():
            # Double dining expenses
            modified_spending["dining"] = current_spending.get("dining", 0) * 2
            value_change = modified_spending["dining"] * 0.03
            recommendation_change = "Card replacement recommended"
        elif "travel" in scenario_change.lower() and "more" in scenario_change.lower():
            # Increase travel expenses
            modified_spending["travel"] = current_spending.get("travel", 0) * 1.5
            value_change = modified_spending["travel"] * 0.04
            recommendation_change = "Strategy adjustment recommended"
        elif "reduc" in scenario_change.lower() and "grocery" in scenario_change.lower():
            # Reduce grocery spending
            modified_spending["groceries"] = current_spending.get("groceries", 0) * 0.7
            value_change = -modified_spending["groceries"] * 0.02
            recommendation_change = "No change recommended"
        else:
            # Default case
            value_change = 0
            recommendation_change = "No change recommended"
        
        # Re-use the existing card strategy but adjust for the modified spending
        if len(current_recommendation.get('top_cards', [])) >= 2:
            card1 = current_recommendation['top_cards'][0]
            card2 = current_recommendation['top_cards'][1]
            
            # Adjust strategy based on scenario
            if "double" in scenario_change.lower() and "dining" in scenario_change.lower():
                # Prioritize dining with card that's better for it
                updated_strategy = {
                    card1['id']: [
                        {"category": "dining", "reason": "Increased rewards for higher dining spend"},
                        {"category": "travel", "reason": "Good for travel expenses"}
                    ],
                    card2['id']: [
                        {"category": "groceries", "reason": "Better rewards for groceries"},
                        {"category": "shopping", "reason": "Good for shopping"}
                    ]
                }
            elif "travel" in scenario_change.lower():
                # Prioritize travel with the miles card
                updated_strategy = {
                    card1['id']: [
                        {"category": "travel", "reason": "Maximized miles for increased travel"},
                        {"category": "transportation", "reason": "Good for transportation"}
                    ],
                    card2['id']: [
                        {"category": "groceries", "reason": "Better for everyday expenses"},
                        {"category": "dining", "reason": "Good for dining"}
                    ]
                }
            else:
                # Default strategy adjustment
                updated_strategy = current_recommendation.get('usage_strategy', {})
                
            return {
                'recommendation_change': recommendation_change,
                'value_change': value_change,
                'updated_strategy': updated_strategy
            }
        else:
            # Not enough cards for a meaningful strategy update
            return {
                'recommendation_change': "No change recommended",
                'value_change': 0.0,
                'updated_strategy': {}
            }
    
    async def search_cards(self, query: str) -> List[Dict[str, Any]]:
        """Search for cards based on a natural language query
        
        Args:
            query: Natural language description of card features to search for
            
        Returns:
            List of matching cards with relevance scores
        """
        await self.initialize()
        return await self.client.search_cards(query)
    
    async def shutdown(self):
        """Shutdown the agent and clean up resources"""
        await self.client.shutdown()
        self.initialized = False


# Demo of the agent
if __name__ == "__main__":
    # Initialize the client and agent
    client = CardOptimizerClient()
    agent = CardOptimizerAgent()
    
    # Sample transactions (preprocessing step)
    transactions = [
        {'merchant': 'NTUC FairPrice', 'amount': 200.50},
        {'merchant': 'Grab Transport', 'amount': 150.75},
        {'merchant': 'McDonald\'s', 'amount': 25.60},
        {'merchant': 'Uniqlo Somerset', 'amount': 120.00},
        {'merchant': 'Netflix Subscription', 'amount': 19.90},
    ]
    
    # Preprocessing: Categorize transactions to create spending profile
    from src.statement_processing.merchant_categorizer import MerchantCategorizer
    categorizer = MerchantCategorizer()
    spending_profile = categorizer.process_transactions(transactions)
    print("Preprocessed Spending Profile:")
    for category, amount in spending_profile.items():
        if amount > 0:
            print(f"- {category}: ${amount:.2f}")
    
    # Sample preferences
    preferences = {
        'type': 'miles',
        'max_annual_fee': 200.00,
        'income': 60000,
        'prefer_airport_lounge': True
    }
    
    # Agent step: Get card recommendations based on preprocessed spending profile
    recommendations = asyncio.run(agent.recommend_cards(spending_profile, preferences))
    print("\nRecommended Cards:")
    for card in recommendations['top_cards']:
        print(f"- {card['name']} (Priority: {card['priority']})")
        
        # Show usage strategy
        if card['id'] in recommendations['usage_strategy']:
            for usage in recommendations['usage_strategy'][card['id']]:
                print(f"  - {usage['category'].capitalize()}: {usage['reason']}")
    
    print(f"\nEstimated Monthly Value: ${recommendations['estimated_value']:.2f}")
    
    # Agent step: Query T&C
    card_id = recommendations['top_cards'][0]['id']
    tc_query = f"What are the miles earning rates for {card_id}?"
    tc_response = asyncio.run(agent.query_terms_and_conditions(tc_query, card_id))
    print(f"\nT&C Query: {tc_query}")
    print(f"Answer: {tc_response['answer']}")
    print(f"Source: {tc_response['source']}")
    
    # Agent step: Scenario analysis
    print("\nScenario Analysis:")
    scenario = "I plan to double my dining expenses next month."
    scenario_result = asyncio.run(agent.analyze_scenario(
        spending_profile,
        scenario,
        recommendations
    ))
    print(f"Scenario: {scenario}")
    print(f"Recommendation: {scenario_result['recommendation_change']}")
    print(f"Value Change: ${scenario_result['value_change']:.2f}")
    
    print("\nUpdated Strategy:")
    for card_id, strategies in scenario_result['updated_strategy'].items():
        card_name = next((c['name'] for c in recommendations['top_cards'] if c['id'] == card_id), card_id)
        print(f"{card_name}:")
        for strategy in strategies:
            print(f"- {strategy['category'].capitalize()}: {strategy['reason']}")
    
    # Agent step: Shutdown
    asyncio.run(agent.shutdown()) 