# Multimodal RAG Assistant

A multimodal LLM-powered RAG assistant that answers questions from PDF, audio, and video files using LangChain, LangGraph, FAISS, and OpenAI models.

## Features

- Ask questions across PDF documents
- Transcribe and search audio files
- Extract and summarize video frames
- Retrieve relevant context using FAISS vector search
- Generate grounded answers with source references and confidence levels

## Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here