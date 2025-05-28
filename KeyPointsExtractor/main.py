from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import shutil
import tempfile
from pathlib import Path
import base64
import uuid
import time
from dotenv import load_dotenv
import json
import logging
import traceback
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# Import the project components
from extractor import KeyPointExtractor
from fast_extractor import FastKeyPointExtractor
from utils import validate_api_keys

# Load environment variables
load_dotenv()

# Performance mode setting
USE_FAST_EXTRACTOR = os.getenv("USE_FAST_EXTRACTOR", "true").lower() == "true"

# Initialize the FastAPI app
app = FastAPI(
    title="Key Points Extractor API",
    description="API for extracting key points from documents and images using advanced OCR and RAG techniques",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a temporary directory for storing uploaded files
TEMP_DIR = Path(tempfile.gettempdir()) / "keypoints_extractor"
TEMP_DIR.mkdir(exist_ok=True)

# Store active extraction jobs
active_jobs = {}

# Track startup time for performance monitoring
startup_time = time.time()

# Models for request/response
class KeyPoint(BaseModel):
    text: str
    source: Optional[str] = None

class KeyPointsResponse(BaseModel):
    job_id: str
    status: str
    key_points: Optional[List[KeyPoint]] = None
    message: Optional[str] = None

# New unified request model
class UnifiedRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    num_points: int = 5
    audience: str = "General audience"
    focus_area: Optional[str] = None
    output_format: str = "bullet"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("key_points_extractor")

# Validate API keys on startup
@app.on_event("startup")
async def startup_event():
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        error_msg = "⚠️ CRITICAL: OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable."
        logger.error(error_msg)
        print(f"\n{error_msg}")
        print("The application will run, but key point extraction will fail without a valid API key.")
        print("Set your API key in the .env file or as an environment variable.\n")
    else:
        # Check if the key is valid (proper format)
        if openai_api_key.startswith("sk-") and len(openai_api_key) > 20:
            logger.info("✅ OpenAI API key found with valid format")
        else:
            logger.warning("⚠️ OpenAI API key found but may have invalid format")
    
    # Check for Tesseract installation
    tesseract_path = os.getenv("TESSERACT_PATH", "tesseract")
    if not shutil.which(tesseract_path):
        logger.warning("⚠️ Tesseract OCR not found in PATH. Image processing may fail.")
        print("\n⚠️ Warning: Tesseract OCR not found in PATH. Image processing may fail.")
        print("Install Tesseract OCR and make sure it's in your PATH or set TESSERACT_PATH in .env file.\n")
    else:
        logger.info(f"✅ Tesseract OCR found at: {shutil.which(tesseract_path)}")

# Clean up temporary files on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# Unified endpoint that can handle both form data and JSON
@app.post("/api/extract", response_model=KeyPointsResponse)
async def extract_key_points_unified(request: Request):
    # Generate a job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Processing unified API request with job_id: {job_id}")
    
    # Initialize job status
    active_jobs[job_id] = {
        "status": "processing",
        "upload_time": time.time()
    }
    
    try:
        # Initialize extractor
        if USE_FAST_EXTRACTOR:
            logger.info("Using FastKeyPointExtractor for improved performance")
            extractor = FastKeyPointExtractor()
        else:
            logger.info("Using standard KeyPointExtractor")
            extractor = KeyPointExtractor()
        
        # Check content type to determine how to process the request
        content_type = request.headers.get("content-type", "")
        logger.info(f"Request content type: {content_type}")
        
        success = False
        num_points = 5
        audience = "General audience"
        focus_area = None
        output_format = "bullet"
        processing_details = "Not processed yet"
        
        if "multipart/form-data" in content_type:
            logger.info("Processing multipart form data")
            try:
                # Handle form data
                form = await request.form()
                logger.debug(f"Form fields: {list(form.keys())}")
                
                # Check for file upload
                if "file" in form and form["file"].filename:
                    file = form["file"]
                    logger.info(f"Processing file: {file.filename}")
                    
                    # Perform basic file validation
                    if not validate_file_type(file.filename):
                        error_msg = f"Unsupported file type: {file.filename}"
                        logger.warning(error_msg)
                        active_jobs[job_id]["status"] = "failed"
                        active_jobs[job_id]["message"] = error_msg
                        return {
                            "job_id": job_id,
                            "status": "failed",
                            "message": error_msg
                        }
                    
                    try:
                        # Save the file to a temporary location
                        file_path = TEMP_DIR / f"{job_id}_{file.filename}"
                        logger.debug(f"Saving file to: {file_path}")
                        
                        # Read file content safely
                        content = await file.read()
                        if not content:
                            raise ValueError("File is empty or could not be read")
                        
                        logger.debug(f"File size: {len(content)} bytes, content type: {file.content_type}")
                        
                        # Write to temporary file
                        with open(file_path, "wb") as buffer:
                            buffer.write(content)
                        
                        # Process the file
                        logger.debug(f"Processing file with size: {len(content)} bytes")
                        try:
                            file_content = open(file_path, "rb").read()
                            if USE_FAST_EXTRACTOR:
                                # Use async method for FastKeyPointExtractor
                                success = await extractor.process_file_async(
                                    file_content,
                                    file.filename
                                )
                            else:
                                # Use sync method for standard extractor
                                success = extractor.process_file(
                                    file_content,
                                    file.filename
                                )
                            processing_details = f"Processed file {file.filename} with size {len(content)} bytes"
                        except Exception as e:
                            logger.error(f"Error in extractor.process_file: {str(e)}")
                            logger.error(traceback.format_exc())
                            raise ValueError(f"Extractor failed to process file: {str(e)}")
                        
                        # Remove the temporary file
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary file: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
                        
                elif "text" in form and form["text"]:
                    text_content = str(form["text"])
                    logger.info(f"Processing text input (length: {len(text_content)})")
                    
                    # Perform basic text quality validation
                    if not validate_text_input(text_content):
                        error_msg = "Text appears to contain OCR errors or unreadable content. Processing will continue but results may be limited."
                        logger.warning(error_msg)
                        active_jobs[job_id]["message"] = error_msg
                    
                    try:
                        if USE_FAST_EXTRACTOR:
                            # Use async method for FastKeyPointExtractor
                            success = await extractor.process_text_async(text_content)
                        else:
                            # Use sync method for standard extractor
                            success = extractor.process_text(text_content)
                        processing_details = f"Processed text input with {len(text_content)} characters"
                    except Exception as e:
                        logger.error(f"Error in extractor.process_text: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise ValueError(f"Extractor failed to process text: {str(e)}")
                    
                elif "url" in form and form["url"]:
                    url_content = str(form["url"])
                    logger.info(f"Processing URL: {url_content}")
                    try:
                        if USE_FAST_EXTRACTOR:
                            # Use async method for FastKeyPointExtractor
                            success = await extractor.process_url_async(url_content)
                        else:
                            success = extractor.process_url(url_content)
                        processing_details = f"Processed URL: {url_content}"
                    except Exception as e:
                        logger.error(f"Error in extractor.process_url: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise ValueError(f"Extractor failed to process URL: {str(e)}")
                    
                else:
                    logger.warning("No input provided in form data")
                    raise HTTPException(status_code=400, detail="No input provided in form data. Please provide a file, text, or URL.")
                
                # Get extraction parameters
                try:
                    num_points = int(form.get("num_points", 5))
                except (TypeError, ValueError):
                    logger.warning(f"Invalid num_points value: {form.get('num_points')}, using default: 5")
                    num_points = 5
                    
                audience = str(form.get("audience", "General audience"))
                focus_area = form.get("focus_area")
                if focus_area:
                    focus_area = str(focus_area)
                output_format = str(form.get("output_format", "bullet"))
            except Exception as e:
                logger.error(f"Error processing form data: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"Error processing form data: {str(e)}")
                
        elif "application/json" in content_type:
            logger.info("Processing JSON data")
            try:
                # Handle JSON input
                body = await request.json()
                logger.debug(f"JSON fields: {list(body.keys())}")
                
                if "text" in body and body["text"]:
                    text_content = str(body["text"])
                    logger.info(f"Processing text input from JSON (length: {len(text_content)})")
                    
                    # Perform basic text quality validation
                    if not validate_text_input(text_content):
                        error_msg = "Text appears to contain OCR errors or unreadable content. Processing will continue but results may be limited."
                        logger.warning(error_msg)
                        active_jobs[job_id]["message"] = error_msg
                    
                    try:
                        if USE_FAST_EXTRACTOR:
                            # Use async method for FastKeyPointExtractor
                            success = await extractor.process_text_async(text_content)
                        else:
                            # Use sync method for standard extractor
                            success = extractor.process_text(text_content)
                        processing_details = f"Processed text input with {len(text_content)} characters"
                    except Exception as e:
                        logger.error(f"Error in extractor.process_text: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise ValueError(f"Extractor failed to process text: {str(e)}")
            
                elif "url" in body and body["url"]:
                    url_content = str(body["url"])
                    logger.info(f"Processing URL from JSON: {url_content}")
                    try:
                        if USE_FAST_EXTRACTOR:
                            # Use async method for FastKeyPointExtractor
                            success = await extractor.process_url_async(url_content)
                        else:
                            success = extractor.process_url(url_content)
                        processing_details = f"Processed URL: {url_content}"
                    except Exception as e:
                        logger.error(f"Error in extractor.process_url: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise ValueError(f"Extractor failed to process URL: {str(e)}")
                    
                else:
                    logger.warning("No input provided in JSON")
                    raise HTTPException(status_code=400, detail="No input provided in JSON. Please provide text or URL.")
                
                # Get extraction parameters
                try:
                    num_points = int(body.get("num_points", 5))
                except (TypeError, ValueError):
                    logger.warning(f"Invalid num_points value: {body.get('num_points')}, using default: 5")
                    num_points = 5
                    
                audience = str(body.get("audience", "General audience"))
                focus_area = body.get("focus_area")
                if focus_area:
                    focus_area = str(focus_area)
                output_format = str(body.get("output_format", "bullet"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing JSON data: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"Error processing JSON data: {str(e)}")
            
        else:
            logger.warning(f"Unsupported media type: {content_type}")
            raise HTTPException(status_code=415, detail="Unsupported media type. Please use multipart/form-data or application/json.")
        
        # Check processing success
        if not success:
            logger.warning(f"Processing was not successful. Details: {processing_details}")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"Error processing input: {processing_details}"
            return {
                "job_id": job_id,
                "status": "failed",
                "message": f"Error processing input: {processing_details}"
            }
        
        # Extract key points
        logger.info(f"Extracting key points with parameters: num_points={num_points}, audience={audience}")
        try:
            if USE_FAST_EXTRACTOR:
                # Use async method for FastKeyPointExtractor
                key_points = await extractor.extract_key_points_async(
                    num_points=num_points,
                    audience=audience,
                    focus_area=focus_area,
                    output_format=output_format
                )
            else:
                # Use sync method for standard extractor
                key_points = extractor.extract_key_points(
                    num_points=num_points,
                    audience=audience,
                    focus_area=focus_area,
                    output_format=output_format
                )
            
            # Log the number of key points extracted
            logger.info(f"Successfully extracted {len(key_points) if key_points else 0} key points")
            
            if not key_points or len(key_points) == 0:
                logger.warning("No key points were extracted")
                active_jobs[job_id]["status"] = "completed"
                active_jobs[job_id]["key_points"] = []
                active_jobs[job_id]["message"] = "Processed successfully but no key points were extracted"
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "key_points": [],
                    "message": "Processed successfully but no key points were extracted"
                }
            
            # Check if any key points contain warnings about OCR errors
            has_ocr_warning = False
            for point in key_points:
                if "warning" in point.get("text", "").lower() and "ocr" in point.get("text", "").lower():
                    has_ocr_warning = True
                    break
                    
            if has_ocr_warning:
                logger.warning("Key points contain OCR warning")
                active_jobs[job_id]["message"] = "Document may contain OCR errors. Results may be limited."
            
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
            logger.error(traceback.format_exc())
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"Error extracting key points: {str(e)}"
            return {
                "job_id": job_id,
                "status": "failed",
                "message": f"Error extracting key points: {str(e)}"
            }
        
        # Update job status and store results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["key_points"] = key_points
        
        # Return the key points
        logger.info(f"Successfully processed job: {job_id}")
        return {
            "job_id": job_id,
            "status": "completed",
            "key_points": key_points
        }
        
    except Exception as e:
        logger.error(f"Error in extract_key_points_unified: {str(e)}")
        logger.error(traceback.format_exc())
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    # Check if job exists
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove the job
    del active_jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}

@app.get("/api/status/{job_id}", response_model=KeyPointsResponse)
async def get_job_status(job_id: str):
    """
    Get the status and results of a job.
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        JSON response with job status and results if available
    """
    # Check if job exists
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get job details
    job = active_jobs[job_id]
    
    # Return the job status
    response = {
        "job_id": job_id,
        "status": job.get("status", "unknown")
    }
    
    # Add key points if available
    if "key_points" in job:
        response["key_points"] = job["key_points"]
    
    # Add error message if available
    if "message" in job:
        response["message"] = job["message"]
    
    return response

# Clean up old jobs periodically (runs in a separate thread)
def cleanup_old_jobs():
    """Remove jobs older than 1 hour"""
    current_time = time.time()
    expired_jobs = []
    
    for job_id, job in active_jobs.items():
        # Check if job is older than 1 hour
        if current_time - job.get("upload_time", current_time) > 3600:
            expired_jobs.append(job_id)
    
    # Remove expired jobs
    for job_id in expired_jobs:
        del active_jobs[job_id]

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "fast_mode": USE_FAST_EXTRACTOR}

# Add performance monitoring endpoint
@app.get("/api/performance")
async def get_performance_stats():
    """
    Get performance statistics for monitoring.
    
    Returns:
        Performance metrics and statistics
    """
    try:
        # Initialize extractor to get stats
        if USE_FAST_EXTRACTOR:
            extractor = FastKeyPointExtractor()
            stats = extractor.get_performance_stats()
            stats["extractor_type"] = "FastKeyPointExtractor"
        else:
            stats = {
                "extractor_type": "KeyPointExtractor",
                "message": "Performance stats not available for standard extractor"
            }
        
        # Add system stats
        stats.update({
            "active_jobs": len(active_jobs),
            "fast_mode_enabled": USE_FAST_EXTRACTOR,
            "server_uptime": time.time() - startup_time if 'startup_time' in globals() else 0
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}")
        return {"error": str(e)}

# Custom exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Exception type: {type(exc)}")
    logger.error(traceback.format_exc())
    
    # Return a cleaner error response
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check server logs for details."}
    )

def extract_key_points_task(job_id: str, file_path: str = None, text: str = None, url: str = None):
    """
    Background task to extract key points from a document.
    
    Args:
        job_id: ID of the job to update
        file_path: Path to the uploaded file (optional)
        text: Raw text input (optional)
        url: URL to process (optional)
    """
    logger.info(f"Starting key point extraction job: {job_id}")
    logger.info(f"Input sources - file_path: {file_path is not None}, text: {text is not None}, url: {url is not None}")
    
    # Initialize extractor
    extractor = KeyPointExtractor()
    
    try:
        # Process the document
        processing_successful = False
        
        if file_path:
            logger.info(f"Processing file: {file_path}")
            processing_successful = extractor.process_file(open(file_path, 'rb').read(), os.path.basename(file_path))
        elif text:
            logger.info(f"Processing text with length: {len(text)}")
            logger.info(f"Text preview: {text[:200]}...")
            processing_successful = extractor.process_text(text)
        elif url:
            logger.info(f"Processing URL: {url}")
            processing_successful = extractor.process_url(url)
        
        if not processing_successful:
            logger.error("Document processing failed")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = "Failed to process document"
            return
        
        # Extract key points
        logger.info("Extracting key points")
        key_points = extractor.extract_key_points(num_points=5)
        
        if key_points and len(key_points) > 0:
            logger.info(f"Successfully extracted {len(key_points)} key points")
            # Log the key points for verification
            for i, kp in enumerate(key_points):
                logger.info(f"Key point {i+1}: {kp.get('text', '')[:100]}...")
            
            # Update job status
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["key_points"] = key_points
        else:
            logger.warning("No key points were extracted")
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["key_points"] = []
            active_jobs[job_id]["message"] = "Processed successfully but no key points were extracted"
    
    except Exception as e:
        logger.error(f"Error extracting key points: {str(e)}")
        logger.error(traceback.format_exc())
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = str(e)
    finally:
        # Clean up temporary file if needed
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except:
                logger.warning(f"Failed to remove temporary file: {file_path}")
        
        logger.info(f"Successfully processed job: {job_id}")

# Run the application
if __name__ == "__main__":
    # Clean up old temp files on startup
    for file in TEMP_DIR.glob("*"):
        try:
            if file.is_file():
                os.unlink(file)
        except:
            pass
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8002))
    
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

# Add helper functions for validation
def validate_file_type(filename: str) -> bool:
    """
    Validate if the file type is supported.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if the file type is supported, False otherwise
    """
    supported_extensions = {
        # Text and document formats
        '.txt', '.pdf', '.doc', '.docx', '.rtf', '.odt',
        '.html', '.htm', '.md', '.csv', '.json',
        # Image formats (will be processed with OCR)
        '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.tif', '.bmp', '.gif'
    }
    
    ext = os.path.splitext(filename.lower())[1]
    return ext in supported_extensions

def validate_text_input(text: str) -> bool:
    """
    Perform basic validation of text input to detect potential OCR errors.
    
    Args:
        text: Text to validate
        
    Returns:
        True if the text appears valid, False if it likely contains OCR errors
    """
    if not text or len(text.strip()) == 0:
        return False
        
    # Very short text is fine
    if len(text) < 50:
        return True
        
    # Check for excessive special characters that might indicate OCR errors
    special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
    if special_char_ratio > 0.3:  # If more than 30% of characters are special
        logger.warning(f"Text contains too many special characters: {special_char_ratio:.2f} ratio")
        return False
        
    # Check for real words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 5:  # If text has fewer than 5 real words, it's likely not meaningful
        logger.warning(f"Text contains too few real words: {len(words)}")
        return False
        
    # Check for repetitive patterns that might indicate OCR errors
    repetitive_patterns = re.findall(r'(.{5,})\1{2,}', text)
    if repetitive_patterns:
        logger.warning(f"Text contains repetitive patterns: {repetitive_patterns[:2]}")
        return False
        
    # Check for reasonable vowel ratio in English text
    letter_counts = {}
    for c in text.lower():
        if c.isalpha():
            letter_counts[c] = letter_counts.get(c, 0) + 1
            
    if letter_counts:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowel_count = sum(letter_counts.get(v, 0) for v in vowels)
        total_letters = sum(letter_counts.values())
        vowel_ratio = vowel_count / total_letters if total_letters > 0 else 0
        
        # Natural language typically has a vowel ratio between 0.2 and 0.5
        if vowel_ratio < 0.15 or vowel_ratio > 0.6:
            logger.warning(f"Text has unusual vowel ratio: {vowel_ratio:.2f}")
            return False
            
    return True 