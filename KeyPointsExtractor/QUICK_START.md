# Quick Start Guide

## Setup

1. Clone or download the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on the `env.template` file:
   ```
   cp env.template .env
   ```
4. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

Run the Streamlit application:
```
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501

## Using the Application

1. **Input Content**: Choose from three input methods:
   - Upload a document (PDF, TXT, HTML)
   - Enter text directly
   - Paste a URL to process its content

2. **Configure Extraction**:
   - Set the number of key points to extract
   - Choose the target audience
   - Specify an optional focus area
   - Select the output format

3. **Extract and Review**:
   - Click the "Extract Key Points" button
   - Review the extracted key points
   - Download the results as TXT or HTML

## Troubleshooting

- If you encounter errors related to document processing, ensure your document is in a supported format (PDF, TXT, HTML).
- For API key errors, verify that your `.env` file contains the correct OpenAI API key.
- If facing memory issues with large documents, try breaking them into smaller chunks or adjusting the chunk size in `document_processor.py`. 