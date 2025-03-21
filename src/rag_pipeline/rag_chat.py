import os
import re
from typing import List, Dict, Any, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from src.chat_agent.llm_config import get_llm, get_chat_prompt

class RAGChat:
    """
    A class for handling RAG-based chat interactions about credit card T&Cs.
    """
    
    def __init__(self, vector_store, temperature: float = 0.0):
        """
        Initialize the RAGChat.
        
        Args:
            vector_store: Vector store containing embedded T&C documents.
            temperature: Temperature setting for the LLM.
        """
        self.llm = get_llm(temperature=temperature)
        self.vector_store = vector_store
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        self.chain = self._create_chain()
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval chain.
        
        Returns:
            ConversationalRetrievalChain instance.
        """
        # Get a standard retriever from the vector store
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create a compressor to extract the most relevant parts of the retrieved documents
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Create a compressed retriever
        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # Create the conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=compressed_retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )
        
        return chain
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a user query.
        
        Args:
            query: User query string.
        
        Returns:
            Dictionary containing the response and source documents.
        """
        # Add system context to the query
        enhanced_query = f"{query}"
        
        # Get response from the chain
        response = self.chain({"question": enhanced_query})
        
        # Extract source information
        sources = []
        for doc in response.get("source_documents", []):
            card_name = doc.metadata.get("card_name", "Unknown Card")
            source_file = os.path.basename(doc.metadata.get("source", "Unknown Source"))
            sources.append({"card_name": card_name, "source": source_file})
        
        return {
            "response": response["answer"],
            "sources": sources
        }
    
    def generate_suggested_questions(self, 
                                    user_preferences: Dict[str, Any],
                                    spending_data: Dict[str, float],
                                    recommended_cards: List[str],
                                    num_questions: int = 4) -> List[str]:
        """
        Generate suggested follow-up questions based on user context.
        
        Args:
            user_preferences: User preference data.
            spending_data: User spending data by category.
            recommended_cards: List of recommended card names.
            num_questions: Number of questions to generate.
        
        Returns:
            List of suggested question strings.
        """
        # Create context for the LLM
        reward_preference = user_preferences.get("reward_preference", "Miles")
        annual_fee_tolerance = user_preferences.get("annual_fee_tolerance", 200)
        
        # Format spending data
        spending_text = "\n".join([f"- {category.title()}: ${amount:.2f}" 
                                   for category, amount in spending_data.items()])
        
        # Format recommended cards
        cards_text = "\n".join([f"- {card}" for card in recommended_cards])
        
        prompt = f"""Based on the following user context, generate {num_questions} helpful and specific follow-up questions 
the user might want to ask about their credit card recommendations or T&Cs.

User preferences:
- Reward preference: {reward_preference}
- Annual fee tolerance: ${annual_fee_tolerance}

User spending:
{spending_text}

Recommended cards:
{cards_text}

The questions should be practical and help the user understand:
1. How to maximize rewards with their specific spending pattern
2. Specific terms and conditions that might impact their rewards
3. Scenario-based questions about changing their spending habits
4. Card-specific benefits or limitations

Generate {num_questions} questions only. Each question should be specific, practical, and directly
relevant to the user's context. Respond with the questions only, no additional text.
"""
        
        # Get response from LLM
        messages = [
            SystemMessage(content="You are a helpful credit card assistant."),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        
        # Parse the response to extract questions
        questions = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line and (line.endswith("?") or "?" in line):
                # Clean up the question (remove leading numbers, dashes, etc.)
                question = re.sub(r"^[0-9.-]+\s*", "", line).strip()
                questions.append(question)
        
        # Ensure we have exactly num_questions
        if len(questions) > num_questions:
            questions = questions[:num_questions]
        
        # If we didn't get enough questions, add some generic ones
        generic_questions = [
            f"How would my rewards change if I double my {max(spending_data, key=spending_data.get)} spend?",
            f"What are the annual fee waiver conditions for {recommended_cards[0] if recommended_cards else 'this card'}?",
            f"Is there a cap on {reward_preference.lower()} for these cards?",
            "Which card is best for overseas spending?",
            "How do these cards compare for dining rewards?"
        ]
        
        while len(questions) < num_questions:
            questions.append(generic_questions[len(questions) % len(generic_questions)])
        
        return questions 