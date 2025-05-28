import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

logger = logging.getLogger(__name__)

class SmartChunker:
    """
    Intelligent document chunking system optimized for different content types.
    """
    
    def __init__(self):
        """Initialize the smart chunker with optimized settings."""
        self.chunk_cache = {}
        
        # Optimized chunk sizes for different content types
        self.chunk_configs = {
            'text': {
                'chunk_size': 800,  # Reduced from default 1000
                'chunk_overlap': 100,
                'separators': ["\n\n", "\n", ". ", " ", ""]
            },
            'technical': {
                'chunk_size': 600,  # Smaller for technical content
                'chunk_overlap': 80,
                'separators': ["\n\n", "\n", ". ", "; ", " ", ""]
            },
            'structured': {
                'chunk_size': 1000,  # Larger for structured content
                'chunk_overlap': 150,
                'separators': ["\n\n", "\n", "• ", "- ", ". ", " ", ""]
            },
            'ocr': {
                'chunk_size': 400,  # Smaller for OCR content (often noisy)
                'chunk_overlap': 60,
                'separators': ["\n\n", "\n", ". ", " ", ""]
            }
        }
    
    def detect_content_type(self, text: str) -> str:
        """
        Detect the type of content to optimize chunking strategy.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Content type string
        """
        # Quick content type detection
        text_lower = text.lower()
        
        # Check for OCR artifacts
        ocr_indicators = [
            len(re.findall(r'[^\w\s]', text)) / len(text) > 0.15,  # High special char ratio
            len(re.findall(r'\b[a-zA-Z]{1,2}\b', text)) > len(text.split()) * 0.3,  # Many short words
            'tesseract' in text_lower or 'ocr' in text_lower
        ]
        
        if any(ocr_indicators):
            return 'ocr'
        
        # Check for technical content
        technical_keywords = [
            'api', 'function', 'class', 'method', 'algorithm', 'database',
            'server', 'client', 'protocol', 'framework', 'library'
        ]
        
        if any(keyword in text_lower for keyword in technical_keywords):
            return 'technical'
        
        # Check for structured content
        structured_indicators = [
            len(re.findall(r'^\s*[-•]\s', text, re.MULTILINE)) > 3,  # Bullet points
            len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE)) > 3,  # Numbered lists
            len(re.findall(r'#{1,6}\s', text)) > 2  # Headers
        ]
        
        if any(structured_indicators):
            return 'structured'
        
        return 'text'  # Default
    
    def chunk_documents_smart(self, documents: List[Document]) -> List[Document]:
        """
        Intelligently chunk documents based on content type.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of optimally chunked documents
        """
        if not documents:
            return []
        
        # Check cache
        doc_hash = self._get_documents_hash(documents)
        if doc_hash in self.chunk_cache:
            logger.info("Using cached chunked documents")
            return self.chunk_cache[doc_hash]
        
        chunked_docs = []
        
        for doc in documents:
            if not doc.page_content or len(doc.page_content.strip()) == 0:
                continue
            
            # Detect content type
            content_type = self.detect_content_type(doc.page_content)
            logger.info(f"Detected content type: {content_type} for document")
            
            # Get appropriate chunking configuration
            config = self.chunk_configs.get(content_type, self.chunk_configs['text'])
            
            # Create text splitter with optimized settings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap'],
                separators=config['separators'],
                length_function=len,
                is_separator_regex=False
            )
            
            # Split the document
            try:
                chunks = text_splitter.split_documents([doc])
                
                # Post-process chunks
                processed_chunks = self._post_process_chunks(chunks, content_type)
                chunked_docs.extend(processed_chunks)
                
                logger.info(f"Document chunked into {len(processed_chunks)} chunks using {content_type} strategy")
                
            except Exception as e:
                logger.error(f"Error chunking document: {str(e)}")
                # Fallback: use original document
                chunked_docs.append(doc)
        
        # Cache the result
        self.chunk_cache[doc_hash] = chunked_docs
        
        logger.info(f"Total chunks created: {len(chunked_docs)}")
        return chunked_docs
    
    def _post_process_chunks(self, chunks: List[Document], content_type: str) -> List[Document]:
        """
        Post-process chunks based on content type.
        
        Args:
            chunks: List of document chunks
            content_type: Type of content
            
        Returns:
            Processed chunks
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Skip very short chunks (likely not useful)
            if len(chunk.page_content.strip()) < 50:
                continue
            
            # Clean up chunk content based on type
            if content_type == 'ocr':
                chunk.page_content = self._clean_ocr_text(chunk.page_content)
            elif content_type == 'technical':
                chunk.page_content = self._clean_technical_text(chunk.page_content)
            
            # Add chunk metadata
            chunk.metadata.update({
                'chunk_id': i,
                'content_type': content_type,
                'chunk_length': len(chunk.page_content)
            })
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text to improve quality.
        
        Args:
            text: OCR text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove isolated single characters (common OCR errors)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        
        # Remove lines with too many special characters
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip():
                special_char_ratio = len(re.findall(r'[^\w\s]', line)) / len(line)
                if special_char_ratio < 0.5:  # Keep lines with reasonable special char ratio
                    cleaned_lines.append(line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def _clean_technical_text(self, text: str) -> str:
        """
        Clean technical text to preserve important elements.
        
        Args:
            text: Technical text to clean
            
        Returns:
            Cleaned text
        """
        # Preserve code blocks and technical terms
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Reduce multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        
        return text.strip()
    
    def _get_documents_hash(self, documents: List[Document]) -> str:
        """
        Generate hash for documents for caching.
        
        Args:
            documents: List of documents
            
        Returns:
            Hash string
        """
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_optimal_chunk_size(self, text: str) -> int:
        """
        Get optimal chunk size for given text.
        
        Args:
            text: Input text
            
        Returns:
            Optimal chunk size
        """
        content_type = self.detect_content_type(text)
        return self.chunk_configs[content_type]['chunk_size']
    
    def clear_cache(self):
        """Clear the chunk cache to free memory."""
        self.chunk_cache.clear()
        logger.info("Chunk cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            'cache_size': len(self.chunk_cache),
            'cache_keys': list(self.chunk_cache.keys())
        } 