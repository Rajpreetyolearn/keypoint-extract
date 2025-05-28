import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import numpy as np
import tempfile
import uuid
from dotenv import load_dotenv
import logging
import traceback
import json
import re
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGEngine:
    """
    RAG (Retrieval Augmented Generation) engine for key point extraction.
    """
    
    def __init__(self, embeddings_model: str = "text-embedding-3-small"):
        """
        Initialize the RAG engine.
        
        Args:
            embeddings_model: The name of the embeddings model to use
        """
        logger.info(f"Initializing RAG engine with embeddings model: {embeddings_model}")
        
        # Create a unique collection name for this session to avoid using existing collections
        self.collection_name = f"keypoints_{uuid.uuid4().hex}"
        logger.info(f"Using unique collection name: {self.collection_name}")
        
        # Initialize embedding model
        try:
            # Use OpenAI embeddings
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
            logger.error(traceback.format_exc())
            raise
            
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,
                openai_api_key=openai_api_key
            )
            logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Configuration for vector store
        self.use_in_memory = False
        logger.info(f"Using in-memory vector store: {self.use_in_memory}")
        
        # Get Qdrant configuration if not using in-memory
        if not self.use_in_memory:
            self.qdrant_url = os.getenv("QDRANT_URL", "https://c6fd972a-3f78-4874-9a48-125d4b2aa249.us-east4-0.gcp.cloud.qdrant.io")
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        
        # Initialize vector store to None (will be created later)
        self.vector_store = None
    
    def _validate_documents_for_qdrant(self, documents: List[Document]) -> None:
        """
        Validate documents to ensure they are compatible with Qdrant.
        
        Args:
            documents: List of LangChain Document objects
        """
        for i, doc in enumerate(documents):
            # Check for boolean values in metadata
            for key, value in doc.metadata.items():
                if isinstance(value, bool):
                    logger.warning(f"Document {i} contains boolean value in metadata key '{key}'. "
                                 "This may cause issues with Qdrant. Consider converting to string.")
                
                # Check for numpy data types
                if str(type(value)).startswith("<class 'numpy"):
                    logger.warning(f"Document {i} contains unsupported type {type(value)} in metadata key '{key}'. "
                                 "This may cause issues with Qdrant. Consider converting to string.")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from the given documents.
        
        Args:
            documents: List of LangChain Document objects
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Validate documents to warn about potential issues
        self._validate_documents_for_qdrant(documents)
        
        try:
            self._create_vector_store_impl(documents)
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            logger.error(traceback.format_exc())
            
            if "memory" not in str(e) and self.use_in_memory is False:
                logger.warning("Falling back to in-memory vector store due to error with hosted instance")
                self.use_in_memory = True
                try:
                    self._create_in_memory_vector_store(documents)
                except Exception as e2:
                    logger.error(f"Fallback to in-memory vector store also failed: {str(e2)}")
                    logger.error(traceback.format_exc())
                    raise Exception(f"Failed to create vector store: {str(e)}. Fallback also failed: {str(e2)}")
            else:
                raise
    
    def _create_vector_store_impl(self, documents: List[Document]) -> None:
        """
        Implementation of vector store creation logic.
        
        Args:
            documents: List of LangChain Document objects
        """
        if self.use_in_memory:
            self._create_in_memory_vector_store(documents)
        else:
            self._create_hosted_vector_store(documents)
    
    def _create_in_memory_vector_store(self, documents: List[Document]) -> None:
        """
        Create an in-memory vector store.
        
        Args:
            documents: List of LangChain Document objects
        """
        collection_name = f"keypoints_{uuid.uuid4().hex[:8]}"
        logger.info(f"Using in-memory vector store with collection name: {collection_name}")
        
        try:
            # Verify documents have page_content set before creating vector store
            for i, doc in enumerate(documents):
                if not doc.page_content or doc.page_content.strip() == "":
                    logger.warning(f"Document at index {i} has empty page_content. Setting default content.")
                    doc.page_content = "Empty document content"
            
            self.vector_store = Qdrant.from_documents(
                documents=documents, 
                embedding=self.embeddings,
                collection_name=collection_name,
                location=":memory:",
                content_payload_key="page_content",  # Explicitly set the content field name
                metadata_payload_key="metadata"      # Explicitly set the metadata field name
            )
            logger.info("In-memory vector store created successfully")
            
            # Verify retrieval works
            self._verify_vector_store_retrieval()
        except Exception as e:
            logger.error(f"Failed to create in-memory vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_hosted_vector_store(self, documents: List[Document]) -> None:
        """
        Create a hosted vector store.
        
        Args:
            documents: List of LangChain Document objects
        """
        logger.info(f"Using hosted Qdrant instance at {self.qdrant_url} with collection {self.collection_name}")
        
        try:
            # First, try to delete the collection if it exists to ensure we're working with fresh data
            try:
                client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
                
                # Check if collection exists
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                # If our collection exists, delete it
                if self.collection_name in collection_names:
                    logger.info(f"Deleting existing collection {self.collection_name} to ensure fresh data")
                    client.delete_collection(collection_name=self.collection_name)
                    
                logger.info("Verified clean collection state")
            except Exception as e:
                logger.warning(f"Failed to check/delete collection: {str(e)}")
                # Continue even if this fails
            
            # Verify documents have page_content set before creating vector store
            for i, doc in enumerate(documents):
                if not doc.page_content or doc.page_content.strip() == "":
                    logger.warning(f"Document at index {i} has empty page_content. Setting default content.")
                    doc.page_content = "Empty document content"
                else:
                    # Log the first 100 characters of each document to ensure we're using the right content
                    logger.info(f"Document {i} content: {doc.page_content[:100]}...")
            
            self.vector_store = Qdrant.from_documents(
                documents=documents, 
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                prefer_grpc=True,
                content_payload_key="page_content",  # Explicitly set the content field name
                metadata_payload_key="metadata",     # Explicitly set the metadata field name
                force_recreate=True                 # Force recreation of the collection
            )
            logger.info("Hosted Qdrant vector store created successfully")
            
            # Verify retrieval works by retrieving one of our documents
            self._verify_vector_store_with_input(documents[0].page_content[:50])
        except Exception as e:
            logger.error(f"Failed to create hosted vector store: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _verify_vector_store_with_input(self, sample_text: str) -> None:
        """
        Verify that the vector store retrieval works correctly by using a sample from our input.
        
        Args:
            sample_text: Sample text from input documents to verify retrieval
        """
        try:
            logger.info(f"Verifying vector store with input text: {sample_text[:30]}...")
            # Use the sample text as query to make sure we get our own documents back
            results = self.vector_store.similarity_search(sample_text, k=1)
            if results:
                logger.info(f"Vector store retrieval verified, found matching document: {results[0].page_content[:100]}...")
            else:
                logger.warning("Vector store retrieval verification returned no results")
        except Exception as e:
            logger.error(f"Vector store retrieval verification failed: {str(e)}")
            logger.error(traceback.format_exc())
            # We don't raise here as this is just a verification step
    
    def _verify_vector_store_retrieval(self) -> None:
        """
        Verify that the vector store retrieval works correctly by attempting a simple query.
        """
        try:
            logger.info("Verifying vector store retrieval capabilities")
            # Use a simple verification query
            results = self.vector_store.similarity_search("test", k=1)
            if results:
                logger.info("Vector store retrieval verification successful")
            else:
                logger.warning("Vector store retrieval verification returned no results")
        except Exception as e:
            logger.error(f"Vector store retrieval verification failed: {str(e)}")
            logger.error(traceback.format_exc())
            # We don't raise here as this is just a verification step
    
    def get_relevant_context(self, query: str, k: int = 5) -> List[Document]:
        """
        Get the most relevant context for a query from the vector store.
        
        Args:
            query: The query to search for
            k: The number of documents to return
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Getting relevant context for query: {query} with k={k}")
        
        if not self.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        try:
            # Perform the similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            
            # Log the content of retrieved documents
            for i, doc in enumerate(docs):
                if hasattr(doc, 'page_content') and doc.page_content:
                    logger.info(f"Document {i} content preview: {doc.page_content[:200]}...")
                else:
                    logger.warning(f"Document {i} has no page_content attribute or empty content")
            
            # Verify and fix any documents with empty page_content
            fixed_docs = []
            for i, doc in enumerate(docs):
                if not hasattr(doc, 'page_content') or not doc.page_content or doc.page_content.strip() == "":
                    logger.warning(f"Retrieved document at index {i} has empty page_content. Using default content.")
                    # Create a new document with default content
                    fixed_doc = Document(
                        page_content="Empty document content",
                        metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                    )
                    fixed_docs.append(fixed_doc)
                else:
                    fixed_docs.append(doc)
                    
            # If all documents were empty, log a warning
            if not fixed_docs:
                logger.warning("No valid documents retrieved from vector store")
                
            return fixed_docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.error(traceback.format_exc())
            # Return an empty list rather than raising an exception
            return []
    
    def extract_key_points(
        self, 
        num_points: int = 5,
        audience: str = "General audience",
        focus_area: Optional[str] = None,
        output_format: str = "bullet"
    ) -> List[Dict[str, str]]:
        """
        Extract key points from documents.
        
        Args:
            num_points: Number of key points to extract
            audience: Target audience for the key points
            focus_area: Optional area to focus on
            output_format: Format of the output (bullet, paragraph, etc.)
            
        Returns:
            List of key points
        """
        logger.info(f"Extracting key points with parameters: num_points={num_points}, audience={audience}, focus_area={focus_area}")
        
        if not self.vector_store:
            error_msg = "Vector store not initialized. Call create_vector_store first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(5, num_points * 2)}  # Get more docs than needed for better context
        )
        logger.info("Created retriever from vector store")
        
        # Use query that will help get good overview of the document
        query = "What is this document about? Summarize the key topics and information."
        logger.info(f"Using query for retrieval: {query}")
        
        try:
            # Get relevant documents
            relevant_docs = retriever.invoke(query)
            
            # Check if we have any valid documents
            if not relevant_docs:
                logger.warning("No documents retrieved for key points extraction")
                return []
                
            # Verify documents have valid content
            valid_docs = []
            for i, doc in enumerate(relevant_docs):
                if hasattr(doc, 'page_content') and doc.page_content and doc.page_content.strip() != "":
                    # Check if the document contains a warning about OCR errors
                    if "warning" in doc.page_content.lower() and "ocr error" in doc.page_content.lower():
                        logger.warning(f"Document {i} contains OCR error warning")
                        # Include this document to ensure the warning is passed to the LLM
                        valid_docs.append(doc)
                        break
                    valid_docs.append(doc)
                    # Add logging to show document content
                    logger.info(f"Document {i} content preview: {doc.page_content[:200]}...")
                else:
                    logger.warning(f"Document {i} has empty content. Skipping.")
                    
            if not valid_docs:
                logger.warning("No valid documents with content available for extraction")
                return [{
                    "text": "No valid content was found to extract key points from.",
                    "source": "system"
                }]
                
            # Create the prompt for LLM
            prompt_text = f"""
            You are an expert at extracting the most important key points from documents, even when those documents contain OCR errors or poorly recognized text.
            
            Based on the following document excerpts, identify the {num_points} most important key points.
            
            Target audience: {audience}
            
            {f'Focus area: {focus_area}' if focus_area else ''}
            
            Document:
            """
            
            # Add document content
            for i, doc in enumerate(valid_docs):
                prompt_text += f"\n--- Document Excerpt {i+1} ---\n{doc.page_content}\n"
                
            prompt_text += f"""
            Now, extract exactly {num_points} key points from these document excerpts.
            Each key point should be a single sentence or short paragraph capturing an important concept or fact.
            
            IMPORTANT INSTRUCTIONS:
            1. NEVER mention OCR errors or document quality in your key points. Focus on extracting actual content.
            2. Even if the document has OCR errors, try to understand the overall theme and extract meaningful information.
            3. Look for patterns, repeated words, or contextual clues to piece together what the document is about.
            4. If you can identify names, dates, locations, or organizations despite OCR errors, include these in your key points.
            5. Look for structural elements like headers, section titles, or formatting that might indicate important content.
            6. If there are parts of the document that are clearly legible, prioritize those for key points.
            7. Focus on actual content, not the quality of the text. Never return system messages about poor quality.
            8. All key points must be marked with source "document", never "system".
            
            Format each key point as a JSON object with two fields:
            - "text": The key point itself (actual content from the document, not comments about document quality)
            - "source": "document" (always use "document" as the source)
            
            Return a valid JSON array containing these key point objects.
            """
            
            # Log the full prompt text being sent to the LLM
            logger.info(f"Prompt for LLM (truncated): {prompt_text[:500]}...")
            
            # Use the LLM to extract key points
            llm_response = self.llm.invoke(prompt_text)
            
            # Handle different response types (AIMessage or string)
            response_text = ""
            if hasattr(llm_response, 'content'):  # AIMessage object
                response_text = llm_response.content
            elif isinstance(llm_response, str):   # String response
                response_text = llm_response
            else:
                # Try to convert to string
                response_text = str(llm_response)
            
            logger.info(f"LLM response type: {type(llm_response)}")
            
            # Extract the JSON part from the response
            pattern = r'\[\s*\{.*\}\s*\]'
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                json_str = match.group(0)
            else:
                # Try to find array with slightly different pattern
                pattern = r'\[\s*\{.*'
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    # Try to complete the JSON if it's incomplete
                    if json_str.count('{') > json_str.count('}'):
                        json_str += '}'
                    if json_str.count('[') > json_str.count(']'):
                        json_str += ']'
                else:
                    # Last resort, try to extract any JSON-like structure
                    json_str = response_text
            
            try:
                # Parse the extracted JSON
                key_points = json.loads(json_str)
                return key_points
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {response_text}")
                # Try a more lenient approach
                try:
                    # Use regex to extract each object and manually create array
                    object_pattern = r'\{\s*"text"\s*:\s*"([^"]*)"\s*,\s*"source"\s*:\s*"([^"]*)"\s*\}'
                    matches = re.findall(object_pattern, response_text)
                    
                    if matches:
                        key_points = [{"text": text, "source": source} for text, source in matches]
                        return key_points[:num_points]  # Limit to requested number
                    else:
                        logger.warning("Could not extract key points from LLM response")
                        return [{
                            "text": "Failed to extract key points from the document. The content may be difficult to process.",
                            "source": "system"
                        }]
                except Exception as e:
                    logger.error(f"Failed to extract key points with fallback method: {str(e)}")
                    return [{
                        "text": "An error occurred while extracting key points.",
                        "source": "system"
                    }]
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            logger.error(traceback.format_exc())
            return [{
                "text": "An error occurred during key point extraction.",
                "source": "system"
            }] 