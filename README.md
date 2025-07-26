# VerseJet 📖✨

A high-performance semantic search engine for Bible verses using OpenAI embeddings and Go.

![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 🚀 Features

- **Semantic Search**: Find verses by meaning, not just keywords
- **Lightning Fast**: In-memory search with <10ms latency
- **OpenAI Integration**: Uses `text-embedding-3-small` for high-quality embeddings
- **Complete KJV**: 31,102+ verses from King James Version
- **Context Aware**: Returns verses with surrounding context
- **Production Ready**: Docker support, health checks, graceful shutdown
- **RESTful API**: Simple JSON API for easy integration

## 🛠️ Quick Start

### Prerequisites

- Go 1.21+
- OpenAI API key
- Docker (optional)

### 1. Clone & Setup

```bash
git clone <repository-url>
cd versejet
cp configs/.env.sample .env
# Edit .env with your OpenAI API key
```

### 2. Build Index (First Time Only)

```bash
# Build the indexer
go build -o indexer cmd/indexer/main.go

# Generate the verse index (~195MB)
./indexer -text verses-1769.json -embeddings VersejetKJV.json -output data/bible-index.gob
```

### 3. Run the Server

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the server
go run main.go
```

The server will start on port 8080.

## 🐳 Docker Usage

### Build and Run

```bash
# Build the image
docker build -t versejet .

# Run with environment variables
docker run -p 8080:8080 -e OPENAI_API_KEY="your-api-key" versejet
```

### Docker Compose

```bash
# Set your API key in .env
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Start the service
docker-compose up -d
```

## 📡 API Usage

### Search Endpoint

```bash
POST /query
Content-Type: application/json

{
  "query": "love your enemies",
  "k": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "ref": "Matthew 5:44",
      "text": "But I say unto you, Love your enemies, bless them that curse you...",
      "next_five": "do good to them that hate you, and pray for them...",
      "score": 0.92
    }
  ],
  "query": "love your enemies",
  "count": 20
}
```

### Health Check

```bash
GET /healthz
```

## 🔧 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | *required* | Your OpenAI API key |
| `PORT` | `8080` | Server port |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `INDEX_PATH` | `data/bible-index.gob` | Path to verse index file |

## 📊 Performance

- **Index Size**: ~195MB (31,102+ verses with 1536-dim embeddings)
- **Memory Usage**: ~200MB runtime
- **Query Latency**: 
  - Embedding generation: 50-150ms (OpenAI API)
  - Similarity search: <10ms (in-memory)
  - Total response: <200ms
- **Throughput**: 1000+ queries/second (limited by OpenAI API)

## 🧪 Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./internal/index
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   HTTP Client   │───▶│  API Handler │───▶│  OpenAI Client  │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Verse Index    │
                    │  (In-Memory)     │
                    │                  │
                    │ • 31K+ verses    │
                    │ • Embeddings     │
                    │ • Cosine Search  │
                    └──────────────────┘
```

### Core Components

- **`cmd/indexer/`**: CLI tool for building the verse index
- **`internal/index/`**: Verse data structures and search engine
- **`internal/api/`**: HTTP handlers and OpenAI integration
- **`main.go`**: Server bootstrap and configuration

## 📁 Project Structure

```
versejet/
├── cmd/
│   └── indexer/           # Index building CLI
├── configs/               # Configuration files
├── data/                  # Generated index files
├── internal/
│   ├── api/              # HTTP handlers
│   └── index/            # Search engine
├── verses-1769.json      # KJV verse text
├── VersejetKJV.json      # Pre-computed embeddings
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Development setup
└── main.go              # Application entry point
```

## 🔍 Example Queries

```bash
# Find verses about love
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "God is love", "k": 5}'

# Search for comfort in difficult times
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "peace in troubled times", "k": 10}'

# Find verses about faith and trust
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "trust in the Lord with all your heart", "k": 15}'
```

## 🛡️ Security Notes

- Never commit your `.env` file or API keys
- Use environment-specific configuration in production
- Consider rate limiting for public deployments
- The application runs as non-root in Docker

## 🚧 Future Enhancements

- [ ] Multiple Bible translations (NIV, ESV, NASB)
- [ ] Verse bookmarking and collections
- [ ] Advanced search filters (book, chapter, testament)
- [ ] Caching layer with Redis
- [ ] Vector database integration (Pinecone, Weaviate)
- [ ] GraphQL API
- [ ] Web frontend interface
- [ ] Mobile app support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📈 Monitoring

### Health Checks

The application provides health check endpoints:

```bash
# Basic health check
curl http://localhost:8080/healthz

# Docker health check (automatically configured)
docker ps  # Shows health status
```

### Metrics

Monitor these key metrics:

- Query response time
- OpenAI API latency
- Memory usage
- Error rates
- Request throughput

## 🔧 Troubleshooting

### Common Issues

**"Failed to load verse index"**
- Ensure `data/bible-index.gob` exists
- Run the indexer to generate the file
- Check file permissions

**"OpenAI API errors"**
- Verify your API key is correct
- Check API quota and billing
- Ensure network connectivity

**"Out of memory"**
- The index requires ~200MB RAM
- Consider reducing the number of verses for testing

### Debug Mode

Enable verbose logging:

```bash
export LOG_LEVEL=debug
go run main.go
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing excellent embedding models
- King James Version text from public domain sources
- Go community for excellent libraries and tools

---

**Built with ❤️ and Go**

For questions or support, please open an issue on GitHub.