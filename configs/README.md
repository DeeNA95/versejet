# VerseJet Configuration

This directory contains configuration files and documentation for the VerseJet Bible verse semantic search application.

## Environment Variables

Create a `.env` file in the root directory with the following variables:

### Required Variables

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Embedding Model (default: text-embedding-3-small)
EMBEDDING_MODEL=text-embedding-3-small
```

### Optional Variables

```bash
# Server Configuration
PORT=8080

# Index File Path
INDEX_PATH=data/bible-index.gob
```

## Configuration Details

### OPENAI_API_KEY
- **Required**: Yes
- **Description**: Your OpenAI API key for generating embeddings
- **How to get**: Visit https://platform.openai.com/api-keys
- **Security**: Never commit this to version control

### EMBEDDING_MODEL
- **Required**: No
- **Default**: `text-embedding-3-small`
- **Description**: OpenAI embedding model to use for queries
- **Options**: 
  - `text-embedding-3-small` (1536 dimensions, faster, cheaper)
  - `text-embedding-3-large` (3072 dimensions, higher quality)
  - `text-embedding-ada-002` (1536 dimensions, legacy)

### PORT
- **Required**: No
- **Default**: `8080`
- **Description**: Port number for the HTTP server

### INDEX_PATH
- **Required**: No
- **Default**: `data/bible-index.gob`
- **Description**: Path to the precomputed verse index file

## Example .env File

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-abc123...

# Server Settings
PORT=8080
EMBEDDING_MODEL=text-embedding-3-small
INDEX_PATH=data/bible-index.gob
```

## Security Notes

1. **Never commit** your `.env` file to version control
2. The `.env` file is already included in `.gitignore`
3. Use environment-specific configuration for production deployments
4. Consider using secrets management systems in production

## Production Deployment

For production environments, set environment variables directly rather than using a `.env` file:

```bash
export OPENAI_API_KEY="your-api-key"
export PORT="8080"
export EMBEDDING_MODEL="text-embedding-3-small"
```

Or use your container orchestration platform's secrets management.