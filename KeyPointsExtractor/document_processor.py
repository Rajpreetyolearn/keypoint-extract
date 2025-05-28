import os
from typing import List, Dict, Any, Optional, Union
import tempfile
from pathlib import Path
import re
import io
import numpy as np
import string
import difflib
from collections import Counter
import logging
import traceback
import shutil

# Document processing libraries
from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
import trafilatura
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2

# LangChain document loaders and text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """
    Class for processing different document types and preparing them for RAG.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Check for Tesseract installation
        self.tesseract_path = os.getenv("TESSERACT_PATH", "tesseract")
        if not shutil.which(self.tesseract_path):
            self.logger.warning("⚠️ Tesseract OCR not found in PATH. Image processing will fail.")
            self.ocr_available = False
        else:
            self.logger.info(f"✅ Tesseract OCR found at: {shutil.which(self.tesseract_path)}")
            self.ocr_available = True
            # Configure pytesseract with the correct path
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = shutil.which(self.tesseract_path)
                self.logger.info("Pytesseract configured successfully")
            except Exception as e:
                self.logger.error(f"Failed to configure pytesseract: {str(e)}")
                self.ocr_available = False
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Common words dictionary for error correction
        self.common_words = [
            "the", "and", "to", "of", "in", "a", "for", "is", "on", "that", "by", "this", "with", "i", "you", "it",
            "not", "or", "be", "are", "from", "at", "as", "your", "all", "have", "new", "more", "an", "was", "we",
            "will", "home", "can", "us", "about", "if", "page", "my", "has", "search", "free", "but", "our", "one",
            "other", "do", "no", "information", "time", "they", "site", "he", "up", "may", "what", "which", "their",
            "news", "out", "use", "any", "there", "see", "only", "so", "his", "when", "contact", "here", "business",
            "who", "web", "also", "now", "help", "get", "pm", "view", "online", "first", "am", "been", "would",
            "how", "were", "me", "services", "some", "these", "click", "its", "like", "service", "than", "find",
            "price", "date", "back", "top", "people", "had", "list", "name", "just", "over", "state", "year", "day",
            "into", "email", "two", "health", "world", "next", "used", "go", "work", "last", "most", "products",
            "music", "buy", "data", "make", "should", "report", "company", "after", "video", "line", "system", "post",
            "her", "city", "add", "policy", "number", "such", "please", "available", "copyright", "support",
            "message", "version", "rights", "public", "school", "through", "each", "links", "review", "years",
            "order", "very", "privacy", "book", "items", "group", "need", "many", "user", "said", "does", "set",
            "under", "general", "research", "university", "january", "mail", "full", "map", "reviews"
        ]
        
        # Common date patterns
        self.date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years like 1999, 2023
            r'\b\d{1,2}/\d{1,2}/(19|20)\d{2}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-(19|20)\d{2}\b',  # MM-DD-YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? (19|20)\d{2}\b',  # Month DD, YYYY
            r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (19|20)\d{2}\b',  # DD Month YYYY
        ]
        
        # Suspicious character patterns
        self.suspicious_patterns = [
            r'[§£¢€¥©®™]',  # Special currency/symbol characters that are often misrecognized
            r'[^\x00-\x7F]+',  # Non-ASCII characters
            r'[A-Z]{5,}',  # All caps words with 5+ characters (often errors)
        ]
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a document based on its file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
        """
        extension = Path(file_path).suffix.lower()
        
        if extension == ".pdf":
            return self.process_pdf(file_path)
        elif extension == ".txt":
            return self.process_text(file_path)
        elif extension in [".html", ".htm"]:
            return self.process_html(file_path)
        elif extension in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif"]:
            return self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangChain Document objects
        """
        pdf_reader = PdfReader(file_path)
        text = ""
        raw_metadata = {
            "source": os.path.basename(file_path),
            "total_pages": len(pdf_reader.pages)
        }
        
        # Extract text with page metadata
        documents = []
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                page_metadata = raw_metadata.copy()
                page_metadata["page"] = i + 1
                # Sanitize metadata
                sanitized_metadata = self.sanitize_metadata(page_metadata)
                doc = Document(page_content=page_text, metadata=sanitized_metadata)
                documents.append(doc)
        
        # Split documents into chunks
        chunked_documents = []
        for doc in documents:
            page_chunks = self.text_splitter.split_documents([doc])
            chunked_documents.extend(page_chunks)
        
        return chunked_documents
    
    def detect_handwriting(self, image):
        """
        Detect if an image contains handwritten text and return a confidence score.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            float: confidence score between 0 and 1 indicating likelihood of handwriting
        """
        try:
            # Convert to numpy array if PIL image
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert('L'))
            else:
                img_np = image
                
            # Apply Canny edge detection
            edges = cv2.Canny(img_np, 50, 150)
            
            # Calculate histogram of oriented gradients (simplified approach)
            # Handwriting typically has more varied gradient orientations
            sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitudes and angles
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            angle = np.arctan2(sobely, sobelx) * 180 / np.pi
            
            # Count gradient angles in different bins
            angle_bins = np.histogram(angle, bins=18, range=(-180, 180))[0]
            magnitude_sum = np.sum(magnitude)
            
            # Normalize the histogram
            if magnitude_sum > 0:
                angle_hist = angle_bins / magnitude_sum
            else:
                angle_hist = angle_bins
                
            # Calculate standard deviation of the histogram
            # Handwriting typically has higher standard deviation
            angle_std = np.std(angle_hist)
            
            # Calculate a handwriting score based on the standard deviation
            # Higher std deviation suggests more varied stroke directions (handwriting)
            # This is a simplified approach and can be refined
            handwriting_confidence = min(1.0, max(0.0, angle_std * 15))
            
            return handwriting_confidence
            
        except Exception:
            # Default to neutral if detection fails
            return 0.5
    
    def preprocess_for_handwriting(self, img):
        """
        Apply specialized preprocessing techniques for handwritten text.
        
        Args:
            img: PIL Image object
            
        Returns:
            List of processed PIL Image objects optimized for handwriting OCR
        """
        processed_images = []
        
        # Convert to grayscale
        img_gray = img.convert('L')
        processed_images.append(img_gray)
        
        # Higher contrast for handwriting
        enhancer = ImageEnhance.Contrast(img_gray)
        img_contrast = enhancer.enhance(2.5)  # Higher contrast for handwriting
        processed_images.append(img_contrast)
        
        # Try multiple thresholding approaches
        try:
            img_np = np.array(img_gray)
            
            # Adaptive thresholding - better for varying backgrounds in handwriting
            adaptive_thresh = cv2.adaptiveThreshold(
                img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(Image.fromarray(adaptive_thresh))
            
            # Adaptive thresholding with different parameters
            adaptive_thresh2 = cv2.adaptiveThreshold(
                img_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 15, 5
            )
            processed_images.append(Image.fromarray(adaptive_thresh2))
            
            # Otsu's threshold with opening to reduce noise
            _, otsu_thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
            processed_images.append(Image.fromarray(opening))
            
        except Exception:
            # Skip if OpenCV processing fails
            pass
        
        # Image dilation - can help connect broken handwriting strokes
        try:
            img_np = np.array(img_gray)
            kernel = np.ones((2, 2), np.uint8)
            dilation = cv2.dilate(img_np, kernel, iterations=1)
            processed_images.append(Image.fromarray(dilation))
        except Exception:
            # Skip if dilation fails
            pass
            
        # Reduce noise while preserving edges (good for handwriting)
        try:
            img_np = np.array(img_gray)
            denoised = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
            processed_images.append(Image.fromarray(denoised))
        except Exception:
            # Skip if denoising fails
            pass
            
        return processed_images
    
    def preprocess_image(self, img):
        """
        Apply various preprocessing techniques to improve OCR accuracy.
        
        Args:
            img: PIL Image object
            
        Returns:
            List of processed PIL Image objects for OCR attempts
        """
        processed_images = []
        
        # Original image
        processed_images.append(img)
        
        # Convert to grayscale
        img_gray = img.convert('L')
        processed_images.append(img_gray)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img_gray)
        img_contrast = enhancer.enhance(2.0)
        processed_images.append(img_contrast)
        
        # Sharpen
        img_sharp = img_gray.filter(ImageFilter.SHARPEN)
        processed_images.append(img_sharp)
        
        # Threshold (binarization)
        try:
            # Convert to numpy array for thresholding
            img_np = np.array(img_gray)
            _, img_thresh_np = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_thresh = Image.fromarray(img_thresh_np)
            processed_images.append(img_thresh)
        except Exception:
            # If cv2 processing fails, skip this step
            pass
        
        # Noise reduction
        img_noise_reduced = img_gray.filter(ImageFilter.MedianFilter(size=3))
        processed_images.append(img_noise_reduced)
        
        return processed_images
    
    def correct_word(self, word, min_length=4):
        """
        Attempt to correct a potentially misspelled word by finding closest match.
        
        Args:
            word: The word to check and potentially correct
            min_length: Minimum word length to attempt correction on
            
        Returns:
            The corrected word or the original if no good match found
        """
        # Don't try to correct short words, punctuation, or numbers
        if len(word) < min_length or word.lower() in self.common_words:
            return word
            
        # Check if the word is mostly digits or special characters
        if sum(c.isdigit() or c in string.punctuation for c in word) > len(word) / 2:
            return word
            
        # Find close matches in our dictionary
        matches = difflib.get_close_matches(word.lower(), self.common_words, n=1, cutoff=0.8)
        
        if matches:
            # Preserve original capitalization
            if word[0].isupper():
                return matches[0].capitalize()
            return matches[0]
        
        return word

    def clean_date_strings(self, text):
        """
        Attempt to standardize or correct date formats in text
        
        Args:
            text: Text potentially containing dates
            
        Returns:
            Text with corrected date formats
        """
        # Find potential invalid years (e.g., 2902)
        def year_replacer(match):
            year = match.group(0)
            if year.isdigit() and len(year) == 4:
                # If first two digits make sense as a century marker (19, 20, 21)
                prefix = year[:2]
                if prefix in ["19", "20", "21"]:
                    return year
                # Otherwise assume it's a typo of a recent year
                return "20" + year[2:]
            return year
            
        # Fix years outside valid ranges
        text = re.sub(r'\b\d{4}\b', year_replacer, text)
        
        return text
    
    def postprocess_ocr_text(self, text, is_handwriting=False):
        """
        Apply various post-processing techniques to improve OCR output quality
        
        Args:
            text: Raw OCR output text
            is_handwriting: Whether the text is from handwritten content
            
        Returns:
            Improved text after post-processing
        """
        if not text or text.isspace():
            return text
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split into lines for processing
        lines = text.split('\n')
        processed_lines = []
        
        # Process symbols that are commonly mistaken
        text = re.sub(r'§', 'S', text)  # Section symbol often confused with S
        text = re.sub(r'£', 'E', text)  # Pound symbol often confused with E
        text = re.sub(r'¢', 'c', text)  # Cent symbol often confused with c
        text = re.sub(r'€', 'e', text)  # Euro symbol often confused with e
        
        # Fix common OCR errors
        replacements = {
            # Common misrecognitions
            'l1': 'll', 'Il': 'll', '0O': '00', 'O0': '00', 
            'rn': 'm', 'cl': 'd', '1I': 'H', 'vv': 'w',
            # Symbols often misrecognized
            '§': 'S', '£': 'E', '¢': 'c', '€': 'e', '¥': 'Y',
            # Words with common errors
            'Tl Ble': 'Table', 'Seelin-w': 'Section', 'Colttane': 'Continue',
            'Senvice': 'Service', 'Commuter': 'Computer', 'Seruice': 'Service',
            'Servlce': 'Service', 'Systen': 'System'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Process each line
        for line in text.split('\n'):
            if line.strip():
                # Clean up dates
                line = self.clean_date_strings(line)
                
                # Word-by-word spelling correction for longer words
                if is_handwriting:
                    words = line.split()
                    corrected_words = [self.correct_word(word) for word in words]
                    line = ' '.join(corrected_words)
                
                processed_lines.append(line)
        
        # Reassemble the text
        processed_text = '\n'.join(processed_lines)
        
        # Look for duplicated words
        processed_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', processed_text)
        
        return processed_text.strip()

    def detect_nonsensical_content(self, text, threshold=0.9):
        """
        Detect if OCR text contains too many likely errors/nonsensical content
        
        Args:
            text: The OCR output text
            threshold: Threshold for determining poor quality (increased from 0.7 to 0.9 to be more permissive)
            
        Returns:
            bool: True if the text appears to be nonsensical/poor quality
        """
        if not text or len(text) < 20:  # Too short to analyze
            return False
            
        # Count words not in our dictionary
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not words:
            return False
            
        unknown_word_ratio = sum(1 for word in words if word not in self.common_words) / len(words)
        
        # Count suspicious patterns
        suspicious_count = 0
        for pattern in self.suspicious_patterns:
            suspicious_count += len(re.findall(pattern, text))
            
        suspicious_ratio = suspicious_count / len(text) if text else 0
        
        # Return true if both ratios are high (more permissive than before)
        return unknown_word_ratio > threshold and suspicious_ratio > 0.2
    
    def process_image(self, file_path: str) -> List[Document]:
        """
        Process an image document using OCR with multiple preprocessing and settings.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of LangChain Document objects
        """
        # First check if OCR is available
        if not self.ocr_available:
            self.logger.error("Cannot process image: Tesseract OCR is not available")
            error_text = "Image processing failed: Tesseract OCR is not available. Please install Tesseract."
            metadata = self.sanitize_metadata({
                "source": os.path.basename(file_path),
                "file_type": "image",
                "error": "Tesseract OCR not available"
            })
            doc = Document(page_content=error_text, metadata=metadata)
            return [doc]
        
        try:
            # Ensure pytesseract is available
            import pytesseract
            self.logger.info(f"Processing image file: {file_path}")
            
            # Open the image
            img = Image.open(file_path)
            self.logger.info(f"Image opened successfully. Size: {img.size}, Mode: {img.mode}")
            
            # Check if the image contains handwriting
            handwriting_confidence = self.detect_handwriting(img)
            self.logger.info(f"Handwriting confidence: {handwriting_confidence}")
            
            # Prepare various OCR configurations
            ocr_configs = [
                '--psm 1',  # Auto page segmentation with OSD
                '--psm 3',  # Default: Full auto page segmentation, but no OSD
                '--psm 4',  # Assume a single column of text of variable sizes
                '--psm 6',  # Assume a single uniform block of text
                '--psm 11',  # Sparse text. Find as much text as possible in no particular order
                '--psm 3 --oem 1',  # LSTM only
                '--psm 3 -l eng+osd'  # English language with OSD
            ]
            
            # Add handwriting-specific configs if handwriting is detected
            if handwriting_confidence > 0.3:
                self.logger.info("Handwriting detected, adding specialized configs")
                handwriting_configs = [
                    '--psm 6 --oem 1',  # LSTM only, single block mode
                    '--psm 8 --oem 1',  # Treat as single word, LSTM only
                    '--psm 13 --oem 1',  # Raw line mode, LSTM only
                    '--psm 6 -l eng+equ',  # Better for mixed text/symbols
                    '--psm 6 -c preserve_interword_spaces=1',  # Preserve spacing in handwriting
                    '--psm 6 -c textord_heavy_nr=1',  # More aggressive noise removal
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!:;()\'"%-+=$',  # Limit characters to improve accuracy
                ]
                ocr_configs.extend(handwriting_configs)
            
            # Preprocess images
            self.logger.info("Preprocessing image for OCR")
            processed_images = self.preprocess_image(img)
            
            # Add handwriting-specific preprocessing if handwriting is detected
            if handwriting_confidence > 0.3:
                self.logger.info("Adding handwriting-specific preprocessing")
                handwriting_images = self.preprocess_for_handwriting(img)
                processed_images.extend(handwriting_images)
            
            # Add extra aggressive preprocessing for difficult images
            try:
                self.logger.info("Adding additional preprocessing techniques")
                # Try more extreme contrast enhancement
                img_gray = img.convert('L')
                enhancer = ImageEnhance.Contrast(img_gray)
                extreme_contrast = enhancer.enhance(3.5)
                processed_images.append(extreme_contrast)
                
                # Try negative image (sometimes works better for certain documents)
                img_np = np.array(img_gray)
                negative = 255 - img_np
                processed_images.append(Image.fromarray(negative))
                
                # Try very aggressive thresholding
                _, extreme_thresh = cv2.threshold(np.array(img_gray), 200, 255, cv2.THRESH_BINARY)
                processed_images.append(Image.fromarray(extreme_thresh))
            except Exception as e:
                self.logger.warning(f"Error during additional preprocessing: {str(e)}")
            
            self.logger.info(f"Created {len(processed_images)} preprocessed versions for OCR")
            
            # Try different combinations of images and configs
            all_texts = []
            raw_texts = []  # Store unprocessed text as fallback
            
            successful_configs = 0
            total_configs = len(processed_images) * len(ocr_configs)
            self.logger.info(f"Attempting {total_configs} OCR combinations")
            
            for i, proc_img in enumerate(processed_images):
                for j, config in enumerate(ocr_configs):
                    try:
                        self.logger.debug(f"Running OCR with image {i+1}/{len(processed_images)}, config {j+1}/{len(ocr_configs)}")
                        text = pytesseract.image_to_string(proc_img, config=config)
                        successful_configs += 1
                        
                        if text and text.strip():
                            # Keep raw text as a fallback
                            raw_texts.append(text)
                            
                            # Post-process each extracted text
                            processed_text = self.postprocess_ocr_text(text, is_handwriting=(handwriting_confidence > 0.3))
                            
                            # Only add if not nonsensical, but be more permissive
                            if processed_text and len(processed_text) > 10 and not self.detect_nonsensical_content(processed_text):
                                all_texts.append(processed_text)
                                self.logger.debug(f"Found good text from image {i+1}, config {j+1} ({len(processed_text)} chars)")
                    except Exception as e:
                        # Log error but continue
                        self.logger.warning(f"OCR error with config {config}: {str(e)}")
                        continue
            
            self.logger.info(f"Completed {successful_configs}/{total_configs} OCR attempts")
            self.logger.info(f"Found {len(all_texts)} good text segments and {len(raw_texts)} raw text segments")
            
            # If we extracted any text, combine and clean it
            if all_texts:
                self.logger.info("Combining and cleaning extracted text")
                # Join all extracted texts
                extracted_text = "\n\n".join(all_texts)
                
                # Remove duplicated lines
                unique_lines = set()
                cleaned_lines = []
                for line in extracted_text.split("\n"):
                    clean_line = line.strip()
                    if clean_line and clean_line not in unique_lines:
                        unique_lines.add(clean_line)
                        cleaned_lines.append(clean_line)
                
                extracted_text = "\n".join(cleaned_lines)
                
                # Final post-processing to ensure quality
                extracted_text = self.postprocess_ocr_text(extracted_text, is_handwriting=(handwriting_confidence > 0.3))
                self.logger.info(f"Final extracted text length: {len(extracted_text)} characters")
            else:
                # Use raw texts as fallback if no processed text is accepted
                if raw_texts:
                    self.logger.warning("No good text found, using raw text fallback")
                    # Take the longest raw text as likely the best one
                    best_raw_text = max(raw_texts, key=len)
                    extracted_text = "Warning: Text quality may be poor. Extracted content:\n\n" + best_raw_text
                    self.logger.info(f"Using fallback text with {len(best_raw_text)} characters")
                else:
                    self.logger.error("No text could be extracted from the image")
                    extracted_text = "No text could be extracted from this image. The image may not contain text, or the text might be too unclear for OCR recognition."
            
            # Add debugging and extraction info to metadata
            raw_metadata = {
                "source": os.path.basename(file_path),
                "file_type": "image",
                "handwriting_detected": handwriting_confidence > 0.3,
                "handwriting_confidence": round(handwriting_confidence, 2),
                "extraction_methods_tried": len(processed_images) * len(ocr_configs),
                "text_found": len(all_texts) > 0 or len(raw_texts) > 0,
                "used_fallback": len(all_texts) == 0 and len(raw_texts) > 0,
                "extracted_content_length": len(extracted_text)
            }
            
            # Sanitize metadata to ensure compatibility with vector stores
            metadata = self.sanitize_metadata(raw_metadata)
            self.logger.info(f"Sanitized metadata: {metadata}")
            
            doc = Document(page_content=extracted_text, metadata=metadata)
            chunks = self.text_splitter.split_documents([doc])
            self.logger.info(f"Created {len(chunks)} document chunks")
            
            return chunks
        except Exception as e:
            # If OCR fails, return a document indicating the failure
            self.logger.error(f"Failed to extract text from image: {str(e)}")
            self.logger.error(traceback.format_exc())
            error_text = f"Failed to extract text from image: {str(e)}"
            metadata = self.sanitize_metadata({
                "source": os.path.basename(file_path),
                "file_type": "image",
                "error": str(e)
            })
            doc = Document(page_content=error_text, metadata=metadata)
            return [doc]
    
    def process_text(self, file_path: str) -> List[Document]:
        """
        Process a text document.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of LangChain Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metadata = {"source": os.path.basename(file_path)}
        doc = Document(page_content=text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        return chunks
    
    def process_html(self, file_path: str) -> List[Document]:
        """
        Process an HTML document.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            List of LangChain Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Use trafilatura for main content extraction
        extracted_text = trafilatura.extract(html_content)
        
        if not extracted_text:
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            extracted_text = soup.get_text()
            # Clean up whitespace
            extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        metadata = {"source": os.path.basename(file_path)}
        doc = Document(page_content=extracted_text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        return chunks
    
    def process_url(self, url: str) -> List[Document]:
        """
        Process content from a URL.
        
        Args:
            url: URL to fetch and process
            
        Returns:
            List of LangChain Document objects
        """
        # Fetch content
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract main content using trafilatura
        extracted_text = trafilatura.extract(response.text)
        
        if not extracted_text:
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            extracted_text = soup.get_text()
            # Clean up whitespace
            extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        metadata = {"source": url}
        doc = Document(page_content=extracted_text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        return chunks
    
    def process_text_input(self, text: str, source: str = "user_input") -> List[Document]:
        """
        Process raw text input from the user.
        
        Args:
            text: Raw text input
            source: Source identifier
            
        Returns:
            List of LangChain Document objects
        """
        self.logger.info(f"Processing text input with length {len(text)} characters from source: {source}")
        # Log preview of the text content
        self.logger.info(f"Text content preview: {text[:200]}...")
        
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.text_splitter.split_documents([doc])
        
        self.logger.info(f"Split into {len(chunks)} chunks")
        # Log preview of the first chunk
        if chunks:
            self.logger.info(f"First chunk preview: {chunks[0].page_content[:100]}...")
        
        return chunks 

    def sanitize_metadata(self, metadata: dict) -> dict:
        """
        Sanitize metadata to ensure all values are compatible with Qdrant.
        Convert boolean values to strings as Qdrant doesn't support boolean values.
        Convert numpy data types to native Python types.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in metadata.items():
            # Convert boolean values to strings
            if isinstance(value, bool):
                sanitized[key] = str(value)
            # Convert numpy data types to native Python types
            elif str(type(value)).startswith("<class 'numpy"):
                # Convert numpy data types to their native Python equivalents
                sanitized[key] = value.item() if hasattr(value, 'item') else float(value)
            # Convert any nested dictionaries
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_metadata(value)
            # Convert any other non-serializable types
            elif not isinstance(value, (str, int, float, list, dict)):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized 