import asyncio
import os
import sys
import httpx
from src.statement_processing.merchant_categorizer import MerchantCategorizer

class CardOptimizerClient:
    """Main client that connects to MCP servers and makes tools available to the agent"""
    
    def __init__(self):
        """Initialize the MCP client and connect to required servers"""
        self.client = None
        self.base_url = "http://localhost:8000"
        self.initialized = False
        
        # Initialize merchant categorizer for reference
        self.merchant_categorizer = MerchantCategorizer(model_path="models/merchant_categorizer")
        
    async def initialize(self):
        """Initialize the client by connecting to servers and getting tools"""
        if self.initialized:
            return
            
        # Create HTTP client
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        
        # Test connection
        try:
            response = await self.client.get("/")
            if response.status_code != 200:
                print(f"Warning: Server responded with status code {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to server: {e}")
            
        self.initialized = True
    
    async def shutdown(self):
        """Disconnect from servers and clean up resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.initialized = False
    
    # Helper methods to access MCP tools
    async def get_available_cards(self):
        """Get all available cards"""
        await self.initialize()
        response = await self.client.post("/api/jsonrpc", json={
            "id": "1",
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": "get_available_cards",
                "arguments": {}
            }
        })
        data = response.json()
        return data.get("result", [])
    
    async def get_card_details(self, card_id):
        """Get details for a specific card"""
        await self.initialize()
        response = await self.client.post("/api/jsonrpc", json={
            "id": "2",
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": "get_card_details",
                "arguments": {"card_id": card_id}
            }
        })
        data = response.json()
        return data.get("result", {})
    
    async def query_tc(self, question, card_id):
        """Query terms and conditions for a specific card"""
        await self.initialize()
        response = await self.client.post("/api/jsonrpc", json={
            "id": "3",
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": "query_tc",
                "arguments": {"question": question, "card_id": card_id}
            }
        })
        data = response.json()
        return data.get("result", {})
    
    async def search_cards(self, query):
        """Search for cards matching a query"""
        await self.initialize()
        response = await self.client.post("/api/jsonrpc", json={
            "id": "4",
            "jsonrpc": "2.0",
            "method": "callTool",
            "params": {
                "name": "search_cards",
                "arguments": {"query": query}
            }
        })
        data = response.json()
        return data.get("result", [])


# Simple demo of the client if run directly
async def main():
    # Create client instance
    client = CardOptimizerClient()
    categorizer = MerchantCategorizer()
    
    try:
        # Initialize the client
        await client.initialize()
        
        # Demonstrate preprocessing with the merchant categorizer
        transactions = [
            {'merchant': 'NTUC FairPrice', 'amount': 200.50},
            {'merchant': 'Grab Transport', 'amount': 150.75},
            {'merchant': 'McDonald\'s', 'amount': 25.60},
            {'merchant': 'Uniqlo Somerset', 'amount': 120.00},
            {'merchant': 'Netflix Subscription', 'amount': 19.90},
        ]
        
        spending_profile = categorizer.process_transactions(transactions)
        print("Processed Spending Profile (via Merchant Categorizer):")
        for category, amount in spending_profile.items():
            if amount > 0:
                print(f"- {category}: ${amount:.2f}")
        
        print("\n--- MCP Tool Demonstration ---")
        
        print("\nAvailable Cards (via MCP):")
        cards = await client.get_available_cards()
        for card in cards:
            print(f"- {card['name']} ({card['type']})")
        
        print("\nDetailed Card Info (via MCP):")
        card_details = await client.get_card_details("citi_premiermiles")
        print(f"Name: {card_details['name']}")
        print(f"Annual Fee: ${card_details['annual_fee']}")
        print(f"Promotion: {card_details['promotion']}")
        
        print("\nT&C Query (via MCP):")
        tc_info = await client.query_tc("What is the annual fee?", "citi_premiermiles")
        print(f"Answer: {tc_info['answer']}")
        print(f"Source: {tc_info['source']}")
        
        print("\nCard Search (via MCP):")
        search_results = await client.search_cards("high miles for dining")
        for result in search_results[:3]:  # Show top 3 results
            print(f"- {result['name']} (Relevance: {result['relevance']:.2f})")
            print(f"  Reason: {result['match_reason']}")
    
    finally:
        # Clean up resources
        await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 