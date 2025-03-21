from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from langchain.schema import HumanMessage, SystemMessage
import os
from langchain.chat_models import ChatOpenAI

from .card_database import CardDatabase
from src.chat_agent.llm_config import get_llm, get_synergy_prompt

# Configure logging
logger = logging.getLogger(__name__)

class SynergyCalculator:
    """
    A class for calculating the optimal credit card combinations based on spending patterns.
    """
    
    def __init__(self, card_db: Optional[CardDatabase] = None):
        """
        Initialize the SynergyCalculator.
        
        Args:
            card_db: CardDatabase instance or None to create a new one.
        """
        self.card_db = card_db or CardDatabase()
        if not self.card_db.loaded:
            self.card_db.load_cards()
            
        self.llm = get_llm(temperature=0.0)
    
    def calculate_rewards(self, spending: Dict[str, float], card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate rewards for a single card based on spending.
        
        Args:
            spending: Dictionary mapping spending categories to amounts.
            card: Card data dictionary.
        
        Returns:
            Dictionary with reward calculation results.
        """
        reward_rates = card.get("reward_rates", {})
        caps = card.get("caps_and_limitations", {})
        reward_type = card.get("reward_type", "points").lower()
        
        # Initialize results
        category_rewards = {}
        total_reward = 0.0
        total_spend = 0.0
        
        # Calculate rewards for each spending category
        for category, amount in spending.items():
            if amount <= 0:
                category_rewards[category] = 0.0
                continue
                
            # Get the rate for this category, default to general rate
            rate = reward_rates.get(category.lower(), reward_rates.get("general", 0.0))
            
            # Apply minimum spend requirement if applicable
            min_spend = caps.get("min_spend", 0.0)
            if min_spend > 0 and sum(spending.values()) < min_spend:
                rate = reward_rates.get("general", 0.0)  # Fallback to general rate
            
            # Calculate category reward
            category_reward = amount * rate
            
            # Apply monthly cap if applicable
            monthly_cap = caps.get("monthly_cap", 0.0)
            if monthly_cap > 0 and category_reward > monthly_cap:
                category_reward = monthly_cap
            
            category_rewards[category] = category_reward
            total_reward += category_reward
            total_spend += amount
        
        # Reward unit based on type
        reward_unit = "miles"
        if "cashback" in reward_type:
            reward_unit = "cashback"
        elif "point" in reward_type:
            reward_unit = "points"
        
        # Convert cashback percentage to dollar amount if applicable
        if reward_unit == "cashback":
            for category in category_rewards:
                category_rewards[category] = (category_rewards[category] / 100) * spending[category]
            total_reward = sum(category_rewards.values())
        
        return {
            "card_id": card.get("id", ""),
            "card_name": card.get("name", ""),
            "reward_type": reward_type,
            "reward_unit": reward_unit,
            "total_reward": total_reward,
            "total_spend": total_spend,
            "category_rewards": category_rewards,
            "annual_fee": card.get("annual_fee", {}).get("amount", 0.0)
        }
    
    def calculate_multi_card_rewards(self, 
                                     spending: Dict[str, float], 
                                     cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate optimal rewards using multiple cards.
        
        Args:
            spending: Dictionary mapping spending categories to amounts.
            cards: List of card data dictionaries.
        
        Returns:
            Dictionary with multi-card reward calculation results.
        """
        # Calculate rewards for each card for each category
        card_rewards = {}
        
        for card in cards:
            card_id = card.get("id", "")
            reward_rates = card.get("reward_rates", {})
            reward_type = card.get("reward_type", "").lower()
            
            card_rewards[card_id] = {
                "card_name": card.get("name", ""),
                "reward_type": reward_type,
                "category_rates": {},
                "optimal_categories": []
            }
            
            # Get reward rate for each category
            for category in spending.keys():
                rate = reward_rates.get(category.lower(), reward_rates.get("general", 0.0))
                card_rewards[card_id]["category_rates"][category] = rate
        
        # For each category, find the card with the highest reward rate
        optimal_allocation = {}
        for category, amount in spending.items():
            if amount <= 0:
                continue
                
            best_card_id = None
            best_rate = -1
            
            for card_id, rewards in card_rewards.items():
                rate = rewards["category_rates"].get(category, 0.0)
                if rate > best_rate:
                    best_rate = rate
                    best_card_id = card_id
            
            if best_card_id:
                card_rewards[best_card_id]["optimal_categories"].append(category)
                optimal_allocation[category] = best_card_id
        
        # Calculate total rewards based on optimal allocation
        total_reward = 0.0
        card_totals = {card_id: 0.0 for card_id in card_rewards.keys()}
        category_breakdown = {}
        
        for category, amount in spending.items():
            if amount <= 0 or category not in optimal_allocation:
                continue
                
            card_id = optimal_allocation[category]
            rate = card_rewards[card_id]["category_rates"].get(category, 0.0)
            reward = amount * rate
            
            # Handle cashback conversion
            if "cashback" in card_rewards[card_id]["reward_type"]:
                reward = (reward / 100) * amount
            
            card_totals[card_id] += reward
            total_reward += reward
            
            category_breakdown[category] = {
                "card_id": card_id,
                "card_name": card_rewards[card_id]["card_name"],
                "amount": amount,
                "reward": reward
            }
        
        # Calculate total annual fees
        total_annual_fee = sum(card.get("annual_fee", {}).get("amount", 0.0) for card in cards)
        
        return {
            "total_reward": total_reward,
            "total_annual_fee": total_annual_fee,
            "card_totals": card_totals,
            "optimal_allocation": optimal_allocation,
            "category_breakdown": category_breakdown,
            "cards": [card.get("name", "") for card in cards]
        }
    
    def find_optimal_card_combination(self, 
                                     spending: Dict[str, float], 
                                     user_preferences: Dict[str, Any],
                                     max_cards: int = 3) -> Dict[str, Any]:
        """
        Find the optimal combination of credit cards based on spending and preferences.
        
        Args:
            spending: Dictionary mapping spending categories to amounts.
            user_preferences: User preference data.
            max_cards: Maximum number of cards to consider in combination.
        
        Returns:
            Dictionary with optimization results or None if no valid combinations are found.
        """
        # Extract user preferences
        reward_preference = user_preferences.get("reward_preference", "Miles").lower()
        annual_fee_tolerance = user_preferences.get("annual_fee_tolerance", 200)
        income_range = user_preferences.get("income_range", "30,000 - 50,000")
        
        # Estimate income from range
        income_estimate = self._estimate_income(income_range)
        
        # Filter cards based on preferences
        filtered_cards = self.card_db.filter_cards(
            reward_type=reward_preference if reward_preference != "no preference" else None,
            annual_fee_max=annual_fee_tolerance,
            income_min=income_estimate
        )
        
        if not filtered_cards:
            # If no cards match the criteria, use all cards
            logger.warning("No cards match user preferences. Using all cards.")
            filtered_cards = self.card_db.get_all_cards()
        
        # If still no cards, return None
        if not filtered_cards:
            logger.error("No cards available in the database.")
            return None
        
        # Calculate rewards for individual cards
        card_results = []
        for card in filtered_cards:
            result = self.calculate_rewards(spending, card)
            card_results.append(result)
        
        # Sort cards by total reward
        card_results.sort(key=lambda x: x["total_reward"], reverse=True)
        
        # Take top N cards for combinations
        top_cards = [self.card_db.get_card(result["card_id"]) for result in card_results[:max_cards*2]]
        
        # Check if we have any valid top cards
        if not top_cards or all(card is None for card in top_cards):
            logger.error("No valid cards found after filtering and reward calculation.")
            return None
        
        # Remove any None values from top_cards
        top_cards = [card for card in top_cards if card is not None]
        
        # Generate combinations of cards (up to max_cards)
        combinations = []
        for i in range(len(top_cards)):
            combinations.append([top_cards[i]])
            
            for j in range(i+1, len(top_cards)):
                if len(combinations) >= 10:  # Limit number of combinations to evaluate
                    break
                    
                combinations.append([top_cards[i], top_cards[j]])
                
                if max_cards >= 3:
                    for k in range(j+1, len(top_cards)):
                        if len(combinations) >= 15:  # Further limit 3-card combinations
                            break
                            
                        combinations.append([top_cards[i], top_cards[j], top_cards[k]])
        
        # If no combinations could be formed, return None
        if not combinations:
            logger.error("No card combinations could be formed.")
            return None
        
        # Calculate rewards for each combination
        combination_results = []
        for combo in combinations:
            try:
                result = self.calculate_multi_card_rewards(spending, combo)
                
                # Check if total annual fee exceeds tolerance
                if result["total_annual_fee"] > annual_fee_tolerance:
                    # Apply a penalty to the score
                    result["adjusted_reward"] = result["total_reward"] * (annual_fee_tolerance / (result["total_annual_fee"] + 1))
                else:
                    result["adjusted_reward"] = result["total_reward"]
                    
                combination_results.append(result)
            except Exception as e:
                logger.error(f"Error calculating rewards for combination: {str(e)}")
                # Continue with next combination
                continue
        
        # If no valid combination results, return None
        if not combination_results:
            logger.error("No valid combination results could be calculated.")
            return None
        
        # Sort combinations by adjusted reward
        combination_results.sort(key=lambda x: x["adjusted_reward"], reverse=True)
        
        # Return the best combination
        best_combo = combination_results[0] if combination_results else None
        
        if best_combo:
            # Find the original card objects for the best combo
            best_cards = []
            for card_name in best_combo["cards"]:
                for card in filtered_cards:
                    if card.get("name", "") == card_name:
                        best_cards.append(card)
                        break
            
            # Get usage strategy from LLM
            try:
                strategy = self._generate_usage_strategy(spending, user_preferences, best_cards, best_combo)
                best_combo["usage_strategy"] = strategy
            except Exception as e:
                logger.error(f"Error generating usage strategy: {str(e)}")
                best_combo["usage_strategy"] = "Use each card for the categories where it offers the highest rewards."
        
        return best_combo
    
    def _estimate_income(self, income_range: str) -> float:
        """
        Estimate income from a range string.
        
        Args:
            income_range: Income range string (e.g., "30,000 - 50,000").
        
        Returns:
            Estimated income.
        """
        try:
            # Parse ranges like "30,000 - 50,000"
            if " - " in income_range:
                parts = income_range.replace(",", "").split(" - ")
                lower = float(parts[0])
                upper = float(parts[1])
                return (lower + upper) / 2
            
            # Parse ranges like "Above 120,000"
            if income_range.lower().startswith("above"):
                value = income_range.replace(",", "").lower().replace("above", "").strip()
                return float(value) * 1.2  # Assume 20% above the threshold
            
            # Parse ranges like "Below 30,000"
            if income_range.lower().startswith("below"):
                value = income_range.replace(",", "").lower().replace("below", "").strip()
                return float(value) * 0.8  # Assume 20% below the threshold
            
        except (ValueError, IndexError):
            pass
        
        # Default fallback
        return 50000.0
    
    def _generate_usage_strategy(self, 
                                spending: Dict[str, float], 
                                user_preferences: Dict[str, Any],
                                cards: List[Dict[str, Any]],
                                combo_result: Dict[str, Any]) -> str:
        """
        Generate a usage strategy for a card combination using LLM.
        
        Args:
            spending: Dictionary mapping spending categories to amounts.
            user_preferences: User preference data.
            cards: List of card data dictionaries.
            combo_result: Combination calculation result.
        
        Returns:
            Usage strategy string.
        """
        try:
            # Format spending data
            spending_text = "\n".join([f"- {category.title()}: ${amount:.2f}" 
                                     for category, amount in spending.items() if amount > 0])
            
            # Format card data
            cards_text = ""
            for card in cards:
                cards_text += f"\n\nCard: {card.get('name', '')}\n"
                cards_text += f"Issuer: {card.get('issuer', '')}\n"
                cards_text += f"Annual Fee: ${card.get('annual_fee', {}).get('amount', 0):.2f}\n"
                cards_text += f"Reward Type: {card.get('reward_type', '')}\n"
                
                # Add reward rates
                cards_text += "Reward Rates:\n"
                for category, rate in card.get("reward_rates", {}).items():
                    if rate > 0:
                        cards_text += f"- {category.title()}: {rate}\n"
                
                # Add caps and limitations
                caps = card.get("caps_and_limitations", {})
                if caps.get("monthly_cap", 0) > 0 or caps.get("min_spend", 0) > 0:
                    cards_text += "Limitations:\n"
                    if caps.get("monthly_cap", 0) > 0:
                        cards_text += f"- Monthly Cap: ${caps.get('monthly_cap', 0):.2f}\n"
                    if caps.get("min_spend", 0) > 0:
                        cards_text += f"- Minimum Spend: ${caps.get('min_spend', 0):.2f}\n"
            
            # Format allocation data
            allocation_text = ""
            for category, card_id in combo_result.get("optimal_allocation", {}).items():
                card_name = None
                for card in cards:
                    if card.get("id", "") == card_id:
                        card_name = card.get("name", "")
                        break
                
                if card_name and spending.get(category, 0) > 0:
                    allocation_text += f"- {category.title()} (${spending.get(category, 0):.2f}): {card_name}\n"
            
            # Create prompt for LLM
            prompt = f"""Generate a clear, practical usage strategy for the following credit card combination based on the user's spending pattern.

User Spending Pattern:
{spending_text}

Recommended Cards:
{cards_text}

Optimal Category Allocation:
{allocation_text}

The strategy should include:
1. Which card to use for which spending category
2. Any minimum spend requirements to be aware of
3. Any caps or limitations to consider
4. Tips to maximize rewards (e.g., timing of certain expenses)
5. Annual fee considerations

Keep the strategy concise, practical, and easy to follow. Focus on actionable advice.
"""
            
            # Log prompt for debugging
            logger.debug(f"Usage strategy prompt: {prompt}")
            
            # Get response from LLM
            system_message = get_synergy_prompt()
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            # Log that we're making the API call
            logger.debug("Making OpenAI API call for usage strategy...")
            
            response = self.llm.invoke(messages)
            
            # Log successful response
            logger.debug("OpenAI API call successful")
            
            return response.content
            
        except Exception as e:
            # Log the error
            logger.error(f"Error generating usage strategy: {str(e)}")
            # Return a simple fallback strategy
            return "Use each card for the categories where it offers the highest rewards. Check each card's terms for minimum spend requirements and caps." 