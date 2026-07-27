"""LangChain QA chain for answer generation"""

import logging
import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class QAChain:
    """LangChain-based QA chain for answer generation"""
    
    DEFAULT_PROMPT = """You are a helpful AI assistant answering questions based on provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question using ONLY the information from the context above
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Include citations in your answer using [Source X] format
4. Be concise and accurate

Answer:"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-5.1",
        temperature: float = 0.0,
        custom_prompt: Optional[str] = None
    ):
        """
        Initialize QA chain
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (default: gpt-5.1)
            temperature: Sampling temperature (0 = deterministic)
            custom_prompt: Custom prompt template
        """
        # Auto-detect Azure or OpenAI
        if api_key and api_key.startswith("gsk_"):
            # Groq fallback
            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=temperature,
                base_url="https://api.groq.com/openai/v1"
            )
        elif os.getenv("AZURE_OPENAI_ENDPOINT"):
            # Azure OpenAI
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key,
                api_version="2024-02-01",
                deployment_name=model_name,
                temperature=temperature
            )
        else:
            # OpenAI
            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature
            )
        
        prompt_template = custom_prompt or self.DEFAULT_PROMPT
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info(f"Initialized QA chain (model={model_name}, temp={temperature})")
    
    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate answer from question and context
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Dict with answer and metadata
        """
        try:
            prompt_text = self.prompt.format(question=question, context=context)
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            return {
                'answer': answer,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer': None,
                'success': False,
                'error': str(e)
            }
