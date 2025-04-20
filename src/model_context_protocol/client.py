import asyncio
import os
import sys
import logging
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, types
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

# Set up logger
logger = logging.getLogger('mcp_client')
logger.setLevel(logging.DEBUG)

class CardOptimizerClient:
    """Main client that connects to MCP servers and makes tools available to the agent"""
    
    def __init__(self):
        """Initialize the MCP client and connect to required servers"""
        self.session = None
        # In Docker Compose, services can connect to each other using the service name as hostname
        # Default to localhost for local development, but override with environment variable if set
        mcp_host = os.environ.get("MCP_HOST", "localhost")
        mcp_port = os.environ.get("MCP_PORT", "8000")
        self.base_url = f"http://{mcp_host}:{mcp_port}"
        self.sse_endpoint = "/sse" 
        self.initialized = False
        self.exit_stack = AsyncExitStack()
        logger.info(f"CardOptimizerClient instance created, will connect to {self.base_url}")
        
    async def initialize(self):
        """Initialize the client by connecting to servers and getting tools
        
        Raises:
            ConnectionError: If unable to connect to the MCP server
            TimeoutError: If connection times out
            RuntimeError: For other initialization errors
        """
        if self.initialized:
            logger.debug("Client already initialized, skipping initialization")
            return
        
        try:    
            # Create MCP Client Session using SSE transport
            # Combine the base URL with the SSE endpoint
            full_url = f"{self.base_url}{self.sse_endpoint}"
            logger.info(f"Connecting to MCP server at {full_url}...")
            
            # Add a timeout to SSE connection attempt
            try:
                # Use AsyncExitStack for managing context managers
                transport = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(sse_client(full_url)),
                    timeout=10.0  # 10 second timeout for connection
                )
                
                self.session = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(ClientSession(*transport)),
                    timeout=5.0  # 5 second timeout for session creation
                )
                
                # Initialize the session
                await asyncio.wait_for(
                    self.session.initialize(),
                    timeout=10.0  # 10 second timeout for initialization
                )
                
                logger.info("Successfully connected to MCP server")
                
                # Verify the session is actually working by checking available tools
                try:
                    # Try to list tools as a simple verification that the connection works
                    await asyncio.wait_for(
                        self.session.list_tools(),
                        timeout=5.0
                    )
                    logger.info("Verified MCP server connection by listing tools")
                except Exception as e:
                    logger.error(f"Failed to verify server connection: {str(e)}")
                    raise ConnectionError(f"Connected to server but failed to verify connection: {str(e)}")
                    
                self.initialized = True
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to MCP server at {full_url}")
                raise TimeoutError(f"Connection to MCP server timed out after 10 seconds")
            
        except (TimeoutError, ConnectionError) as e:
            # Re-raise these specific exceptions
            raise
            
        except Exception as e:
            logger.error(f"Error establishing connection: {str(e)}")
            raise ConnectionError(f"Failed to connect to MCP server: {str(e)}")
            
        # If we get here with any errors, clean up
        if not self.initialized:
            await self.shutdown()
            raise RuntimeError("Failed to initialize client due to unknown error")
    
    async def shutdown(self):
        """Disconnect from servers and clean up resources"""
        logger.info("Shutting down MCP client")
        self.initialized = False
        self.session = None
        
        # Close all context managers managed by AsyncExitStack
        await self.exit_stack.aclose()
        
        # Create a new exit stack for potential future connections
        self.exit_stack = AsyncExitStack()
        logger.info("MCP client shutdown complete")
    
    # Helper methods to access MCP tools
    async def get_tools(self):
        """Get all available tools in a format compatible with langchain
        
        Raises:
            ConnectionError: If MCP session is not initialized
            RuntimeError: If tool loading fails
        """
        
        if not self.session:
            msg = "MCP session not initialized properly. Cannot get tools."
            logger.warning(msg)
            raise ConnectionError(msg)
            
        try:
            # Load MCP tools using the langchain-mcp-adapters
            logger.info("Loading tools from MCP server")
            tools = await asyncio.wait_for(
                load_mcp_tools(self.session),
                timeout=15.0  # 15 second timeout for loading tools
            )
            
            if not tools:
                msg = "No tools were loaded from the MCP server"
                logger.warning(msg)
                return []
                
            logger.info(f"Successfully loaded {len(tools)} tools: {[tool.name for tool in tools]}")
            return tools
        except asyncio.TimeoutError:
            msg = "Timeout while loading tools from MCP server"
            logger.error(msg)
            raise TimeoutError(msg)
        except Exception as e:
            msg = f"Error loading MCP tools: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg)
    
    # Helper method to call MCP tools
    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        """Call a specific tool with provided arguments"""
        if not self.session:
            logger.warning("MCP session not initialized properly. Cannot call tools.")
            return None
        
        try:
            # Log the tool call
            logger.info(f"Calling tool '{name}' with arguments: {arguments}")
            
            # Call the tool with provided arguments and set a timeout
            start_time = asyncio.get_event_loop().time()
            result = await asyncio.wait_for(
                self.session.call_tool(name, arguments),
                timeout=30.0  # 30 second timeout
            )
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Log successful response
            logger.info(f"Tool '{name}' call completed in {elapsed:.2f}s with content length: {len(result.content) if result.content else 0}")
            logger.debug(f"Tool '{name}' response content: {result.content[:100]}...")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout error calling tool '{name}' with arguments {arguments}")
            # Return a default response instead of None to avoid breaking the conversation flow
            return types.ToolResponse(
                content=[types.ContentObject(text=f"Tool execution timed out. Please try again with simpler parameters.")],
                metadata={"error": "timeout"}
            )
        except Exception as e:
            logger.error(f"Error calling tool '{name}': {str(e)}")
            # Return a default response with the error message
            return types.ToolResponse(
                content=[types.ContentObject(text=f"Tool execution failed: {str(e)}")],
                metadata={"error": str(e)}
            )


# Simple demo of the client if run directly
async def main():
    # Create client instance
    client = CardOptimizerClient()
    
    try:
        # Initialize the client
        await client.initialize()

        print("\n--- MCP Tool Demonstration ---")
        print(f"Using MCP server at: {client.base_url}{client.sse_endpoint}")
        
        # Get tools
        tools = await client.get_tools()
        print("\n===== Available Tools =====")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
            print(f"   Description: {tool.description}")
            print()

        # Call get_available_cards tool
        print("\n===== Demonstrating: get_available_cards =====")
        result = await client.call_tool("get_available_cards", {})
        print("Available Cards:")
        for i, card in enumerate(result.content, 1):
            print(f"{i}. {card.text}")

        # Call get_card_details tool
        print("\n===== Demonstrating: get_card_details =====")
        card_id = "DBS Vantage Visa Infinite Card"
        print(f"Getting details for: {card_id}")
        result = await client.call_tool("get_card_details", {"card_id": card_id})
        print("Card Details:")
        print("-" * 50)
        print(result.content[0].text)
        print("-" * 50)

        # Call query_tc tool
        print("\n===== Demonstrating: query_tc =====")
        card_id = "DBS Vantage Visa Infinite Card"
        question = "What are the terms and conditions for the DBS Vantage Visa Infinite Card?"
        print(f"Card: {card_id}")
        print(f"Question: {question}")
        result = await client.call_tool("query_tc", {"card_id": card_id, "question": question})
        print("Answer:")
        print("-" * 50)
        print(result.content[0].text)
        print("-" * 50)

        # Call search_cards tool
        print("\n===== Demonstrating: search_cards =====")
        query = "miles rewards with max annual fee of 455, no airport lounge"
        print(f"Search Query: {query}")
        result = await client.call_tool("search_cards", {"query": query, "num_candidates": 5})
        print("Search Results:")
        for i, card in enumerate(result.content, 1):
            print(f"\nResult {i}:")
            print("-" * 30)
            print(card.text)
            print("-" * 30)

    
    finally:
        # Clean up resources
        await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 