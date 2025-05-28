import os
import tempfile
from typing import Optional, List, Dict, Any
import base64
from pathlib import Path

def get_temp_file_path(file_content, file_name: str) -> str:
    """
    Save uploaded file content to a temporary file and return its path.
    
    Args:
        file_content: Content of the uploaded file
        file_name: Name of the file
        
    Returns:
        Path to the temporary file
    """
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    return file_path

def get_file_extension(file_name: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_name: Name of the file
        
    Returns:
        File extension in lowercase
    """
    return Path(file_name).suffix.lower().lstrip(".")

def create_download_link(content: str, filename: str, text: str) -> str:
    """
    Create a download link for text content.
    
    Args:
        content: Text content to download
        filename: Name of the download file
        text: Text to display for the download link
        
    Returns:
        HTML for a download link
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_key_points(key_points: List[Dict[str, Any]], format_type: str = "bullet") -> str:
    """
    Format key points based on the specified format type.
    
    Args:
        key_points: List of key point dictionaries
        format_type: Format type (bullet, numbered, etc.)
        
    Returns:
        Formatted string of key points
    """
    result = ""
    
    if format_type == "bullet":
        for point in key_points:
            result += f"• {point['text']}\n"
            if 'source' in point and point['source']:
                result += f"  Source: {point['source']}\n"
            result += "\n"
    elif format_type == "numbered":
        for i, point in enumerate(key_points, 1):
            result += f"{i}. {point['text']}\n"
            if 'source' in point and point['source']:
                result += f"   Source: {point['source']}\n"
            result += "\n"
    elif format_type == "hierarchical":
        # Assuming key_points have 'level' attribute for hierarchy
        for point in key_points:
            indent = "  " * (point.get('level', 0))
            result += f"{indent}• {point['text']}\n"
            if 'source' in point and point['source']:
                result += f"{indent}  Source: {point['source']}\n"
            result += "\n"
    else:
        # Default to simple text
        for point in key_points:
            result += f"{point['text']}\n\n"
    
    return result

def validate_api_keys() -> bool:
    """
    Validate that necessary API keys are set.
    
    Returns:
        True if all required API keys are set, False otherwise
    """
    required_keys = ["OPENAI_API_KEY"]
    
    for key in required_keys:
        if not os.environ.get(key):
            return False
    
    return True 