# Key Point Extractor RAG Agent

This application extracts key points from documents using a Retrieval Augmented Generation (RAG) approach.

## Features

- Upload various document formats (PDF, TXT, HTML)
- Extract key points based on user preferences
- Customize extraction by target audience, number of points, and more
- Download results in multiple formats

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `document_processor.py`: Document parsing and preprocessing 
- `rag_engine.py`: RAG implementation with embedding and retrieval
- `extractor.py`: Key point extraction logic
- `utils.py`: Utility functions 