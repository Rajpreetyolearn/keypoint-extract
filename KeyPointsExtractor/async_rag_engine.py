import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
import numpy as np
import tempfile
import uuid
from dotenv import load_dotenv
import logging
import traceback
import json
import re
import time
from qdrant_client import QdrantClient
import hashlib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AsyncRAGEngine:
    """
    Async RAG (Retrieval Augmented Generation) engine for faster key point extraction.
    """
    
    def __init__(self, embeddings_model: str = "text-embedding-3-small"):
        """
        Initialize the async RAG engine.
        
        Args:
            embeddings_model: The name of the embeddings model to use
        """
        logger.info(f"Initializing Async RAG engine with embeddings model: {embeddings_model}")
        
        # Create a unique collection name for this session
        self.collection_name = f"keypoints_{uuid.uuid4().hex}"
        logger.info(f"Using unique collection name: {self.collection_name}")
        
        # Initialize embedding model
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                
            self.embeddings = OpenAIEmbeddings(
                model=embeddings_model,
                openai_api_key=openai_api_key
            )
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
            
        # Initialize LLM with async support
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Using faster model for better performance
                temperature=0.0,
                openai_api_key=openai_api_key,
                max_retries=2,  # Reduce retries for faster failure
                request_timeout=30  # Shorter timeout
            )
            logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
        
        # Configuration for vector store
        self.use_in_memory = True  # Use in-memory for faster processing
        logger.info(f"Using in-memory vector store: {self.use_in_memory}")
        
        # Initialize vector store to None
        self.vector_store = None
        
        # Cache for processed documents
        self.document_cache = {}
        self.result_cache = {}
    
    def _get_document_hash(self, documents: List[Document]) -> str:
        """
        Generate a hash for a list of documents for caching.
        
        Args:
            documents: List of documents
            
        Returns:
            Hash string
        """
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    async def create_vector_store_async(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents asynchronously.
        
        Args:
            documents: List of LangChain Document objects
        """
        start_time = time.time()
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Check cache first
        doc_hash = self._get_document_hash(documents)
        if doc_hash in self.document_cache:
            logger.info("Using cached vector store")
            self.vector_store = self.document_cache[doc_hash]
            return
        
        try:
            # Validate and prepare documents
            for i, doc in enumerate(documents):
                if not doc.page_content or doc.page_content.strip() == "":
                    logger.warning(f"Document at index {i} has empty page_content. Setting default content.")
                    doc.page_content = "Empty document content"
            
            # Create vector store in memory for speed
            collection_name = f"keypoints_{uuid.uuid4().hex[:8]}"
            
            # Use asyncio to run the blocking operation in a thread pool
            loop = asyncio.get_event_loop()
            self.vector_store = await loop.run_in_executor(
                None,
                self._create_vector_store_sync,
                documents,
                collection_name
            )
            
            # Cache the result
            self.document_cache[doc_hash] = self.vector_store
            
            elapsed_time = time.time() - start_time
            logger.info(f"Vector store created successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def _create_vector_store_sync(self, documents: List[Document], collection_name: str):
        """
        Synchronous vector store creation (run in thread pool).
        """
        return Qdrant.from_documents(
            documents=documents, 
            embedding=self.embeddings,
            collection_name=collection_name,
            location=":memory:",
            content_payload_key="page_content",
            metadata_payload_key="metadata"
        )
    
    async def get_relevant_context_async(self, query: str, k: int = 5) -> List[Document]:
        """
        Get relevant context asynchronously.
        
        Args:
            query: The query to search for
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized. Call create_vector_store_async first.")
        
        try:
            # Run similarity search in thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None,
                self.vector_store.similarity_search,
                query,
                k
            )
            
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def extract_key_points_async(
        self, 
        num_points: int = 5,
        audience: str = "General audience",
        focus_area: Optional[str] = None,
        output_format: str = "bullet"
    ) -> List[Dict[str, str]]:
        """
        Extract key points asynchronously.
        
        Args:
            num_points: Number of key points to extract
            audience: Target audience for the key points
            focus_area: Optional area to focus on
            output_format: Format of the output
            
        Returns:
            List of key points
        """
        start_time = time.time()
        logger.info(f"Extracting key points async with parameters: num_points={num_points}, audience={audience}")
        
        # Check result cache
        cache_key = f"{num_points}_{audience}_{focus_area}_{output_format}_{self._get_document_hash([])}"
        if cache_key in self.result_cache:
            logger.info("Using cached extraction results")
            return self.result_cache[cache_key]
        
        if not self.vector_store:
            error_msg = "Vector store not initialized. Call create_vector_store_async first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Get relevant documents asynchronously
            query = "What is this document about? Summarize the key topics and information."
            relevant_docs = await self.get_relevant_context_async(query, min(5, num_points * 2))
            
            if not relevant_docs:
                logger.warning("No documents retrieved for key points extraction")
                return []
            
            # Filter valid documents
            valid_docs = [doc for doc in relevant_docs if doc.page_content and doc.page_content.strip()]
            
            if not valid_docs:
                logger.warning("No valid documents with content available for extraction")
                return [{
                    "text": "No valid content was found to extract key points from.",
                    "source": "system"
                }]
            
            # Create optimized prompt for faster processing
            prompt_text = self._create_optimized_prompt(valid_docs, num_points, audience, focus_area)
            
            # Use async LLM call
            response = await self._call_llm_async(prompt_text)
            
            # Parse response
            key_points = self._parse_llm_response(response, num_points)
            
            # Cache the result
            self.result_cache[cache_key] = key_points
            
            elapsed_time = time.time() - start_time
            logger.info(f"Key points extracted successfully in {elapsed_time:.2f} seconds")
            
            return key_points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            return [{
                "text": "An error occurred during key point extraction.",
                "source": "system"
            }]
    
    def _create_optimized_prompt(self, docs: List[Document], num_points: int, audience: str, focus_area: Optional[str]) -> str:
        """
        Create an optimized prompt for faster LLM processing.
        """
        # Limit document content for faster processing
        max_content_length = 2000  # Reduced from unlimited
        combined_content = ""
        
        for i, doc in enumerate(docs[:3]):  # Limit to 3 documents
            content = doc.page_content[:max_content_length//len(docs)]
            combined_content += f"Document {i+1}: {content}\n\n"
        
        prompt = f"""Extract {num_points} key points from the following documents for {audience}.
{f'Focus on: {focus_area}' if focus_area else ''}

Documents:
{combined_content}

Return ONLY a JSON array of objects with "text" and "source" fields. No explanations.
Example: [{{"text": "Key point 1", "source": "document"}}, {{"text": "Key point 2", "source": "document"}}]
"""
        return prompt
    
    async def _call_llm_async(self, prompt: str) -> str:
        """
        Call LLM asynchronously.
        """
        try:
            # Use async invoke
            response = await self.llm.ainvoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str, num_points: int) -> List[Dict[str, str]]:
        """
        Parse LLM response efficiently.
        """
        try:
            # Try to extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                key_points = json.loads(json_str)
                
                # Ensure all points have correct source
                for point in key_points:
                    if point.get("source") != "document":
                        point["source"] = "document"
                
                return key_points[:num_points]
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
        
        # Fallback: create generic key points
        return [
            {"text": f"Key point {i+1} extracted from the document.", "source": "document"}
            for i in range(num_points)
        ]
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.document_cache.clear()
        self.result_cache.clear()
        logger.info("Caches cleared") 