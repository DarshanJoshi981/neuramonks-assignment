# Advanced RAG using LLM

This is a Streamlit web application for performing advanced question answering using the Retrieval-Augmented Generation (RAG) model, powered by a Language Model (LLM) from Hugging Face. It allows users to upload PDF or CSV files, generate embeddings from the text data, and then interactively ask questions related to the uploaded content.

## Features

- Upload PDF or CSV files for processing.
- Generates embeddings from the uploaded text data.
- Provides an interactive chat interface to ask questions about the uploaded content.
- Uses the RAG model to retrieve and generate answers to user questions.

## Setup

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/DarshanJoshi981/neuramonks-assignment.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your_repository
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload a PDF or CSV file using the sidebar file uploader.
3. Ask questions related to the uploaded content in the chat interface.
4. View the generated answers in the conversation history.

## File Structure

- `app.py`: Main Streamlit application script.
- `langchain/`: Python package containing modules for language processing and interaction with the LLM model.
- `data/`: Directory to store uploaded files and generated embeddings.
- `requirements.txt`: List of Python dependencies required for the project.

## Acknowledgments

- This project utilizes the Hugging Face library for natural language processing tasks.
- Streamlit is used for building the interactive web interface.
- The LangChain library provides additional functionality for text processing and interaction with language models.
