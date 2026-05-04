# Multimodal RAG Assistant

A multimodal LLM-powered RAG (Retrieval-Augmented Generation) assistant that answers questions from PDF, audio, and video files. Built with LangChain, LangGraph, FAISS, and OpenAI models.

## Overview

This project implements an intelligent document and media retrieval system that processes multiple file types and provides contextually relevant answers with source references. It leverages vector embeddings for semantic search and large language models for natural language understanding and generation.

## Features

- Query PDF documents with semantic search
- Transcribe and process audio files
- Extract and analyze video frames
- Vector-based retrieval using FAISS
- Grounded answer generation with source citations
- Confidence scoring for responses

## Tech Stack

- LangChain: LLM orchestration and chains
- LangGraph: Workflow automation
- FAISS: Vector similarity search
- OpenAI: Language models and embeddings
- Python: Core implementation

## Getting Started

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Shuang-Lin-Chen/multimodal-rag-assistant.git
cd multimodal-rag-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```
