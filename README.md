# Confluence Q&A Search System

A powerful system that crawls Confluence spaces, extracts Q&A content, and provides an intelligent search interface with AI-powered enhancements.

## Features

### 1. Confluence Content Crawling
- Automated crawling of Confluence spaces
- Support for multiple content patterns (FAQ sections, tables, expandable sections)
- Intelligent Q&A pair extraction
- HTML sanitization and error handling
- Page reference tracking

### 2. Smart Caching
- SQLite-based persistent cache
- Memory-based LRU caching
- Cache invalidation based on page modifications
- Configurable TTL (Time To Live)
- Space-specific caching

### 3. Vector Search
- Semantic search using sentence transformers
- FAISS vector store for efficient similarity search
- Cached vector embeddings
- Configurable number of results
- Similarity score ranking

### 4. AI Enhancement
- LLM-powered answer enhancement using GPT-4/3.5
- Answer validation and quality checks
- Context-aware improvements
- Markdown formatting support
- Batch processing capabilities

### 5. Content Categorization
- Rule-based categorization
- ML-based topic clustering
- Complexity assessment
- Sentiment analysis
- Category distribution tracking

### 6. Statistics & Logging
- Detailed extraction statistics
- Processing time tracking
- Error reporting
- Category distribution analysis
- Success/failure rate monitoring

### 7. Modern Web Interface
- Clean, responsive design using Tailwind CSS
- Real-time search with suggestions
- Enhanced/Original answer toggle
- Settings management
- Loading indicators and error handling

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Confluence account with API access
- OpenAI API key
- Git

### Installation

1. Clone the repository: