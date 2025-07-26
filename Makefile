# VerseJet Makefile
# Development workflow automation

.PHONY: help build test run clean docker deps indexer check-env build-c

# Default target
help: ## Show this help message
	@echo "VerseJet Development Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Environment variables
BINARY_NAME=versejet
INDEXER_BINARY=indexer
DOCKER_IMAGE=versejet:latest
GO_FILES=$(shell find . -name "*.go")

TARGET_ARCH ?= amd64

# Build C library object file and static library
internal/hnsw/csrc/libvector_search.a: internal/hnsw/csrc/vector_search.c internal/hnsw/csrc/vector_search.h
	@mkdir -p internal/hnsw/csrc
	$(CC) -c -fPIC -O2 -fomit-frame-pointer -o internal/hnsw/csrc/vector_search.o internal/hnsw/csrc/vector_search.c
	ar rcs internal/hnsw/csrc/libvector_search.a internal/hnsw/csrc/vector_search.o

.PHONY: build-c
build-c: internal/hnsw/csrc/libvector_search.a

# Build targets
build: build-c ## Build the main application with C lib
	@echo "🔨 Building $(BINARY_NAME)..."
	@env CGO_LDFLAGS="-Linternal/hnsw/csrc -lvector_search -lm" go build -ldflags="-s -w" -o $(BINARY_NAME) main.go
	@echo "✅ Build complete: $(BINARY_NAME)"

build-indexer: build-c ## Build the indexer CLI tool
	@echo "🔨 Building $(INDEXER_BINARY)..."
	@env CGO_LDFLAGS="-Linternal/hnsw/csrc -lvector_search -lm" go build -ldflags="-s -w" -o $(INDEXER_BINARY) cmd/indexer/main.go
	@echo "✅ Build complete: $(INDEXER_BINARY)"

build-all: build build-indexer ## Build both main app and indexer

# Development targets
deps: ## Download and tidy dependencies
	@echo "📦 Downloading dependencies..."
	@go mod download
	@go mod tidy
	@echo "✅ Dependencies updated"

run: check-env build ## Run the application locally
	@echo "🚀 Starting VerseJet server..."
	@./$(BINARY_NAME)

run-dev: check-env ## Run with development settings
	@echo "🚀 Starting VerseJet in development mode..."
	@PORT=8080 LOG_LEVEL=debug ./$(BINARY_NAME)

# Index management
indexer: build-indexer ## Build and run the indexer
	@echo "📚 Building verse index..."
	@./$(INDEXER_BINARY) -text verses-1769.json -embeddings VersejetKJV_recreated.json -output data/bible-index.gob -verbose
	@echo "✅ Index build complete"

check-index: ## Check if index file exists
	@if [ -f "data/bible-index.gob" ]; then \
		echo "✅ Index file exists (size: $$(du -h data/bible-index.gob | cut -f1))"; \
	else \
		echo "❌ Index file missing. Run 'make indexer' first."; \
		exit 1; \
	fi

# Testing targets
test: ## Run all tests
	@echo "🧪 Running tests..."
	@go test ./...
	@echo "✅ All tests passed"

test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)..."
	@go test -v ./...

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	@go test -cover ./...
	@go test -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "✅ Coverage report generated: coverage.html"

benchmark: ## Run benchmark tests
	@echo "📊 Running benchmarks..."
	@go test -bench=. -benchmem ./internal/index
	@echo "✅ Benchmarks complete"

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	@docker build -t $(DOCKER_IMAGE) .
	@echo "✅ Docker image built: $(DOCKER_IMAGE)"

docker-run: docker-build check-env ## Build and run Docker container
	@echo "🐳 Running Docker container..."
	@docker run --rm -p 8080:8080 \
		-e OPENAI_API_KEY="$(OPENAI_API_KEY)" \
		-e EMBEDDING_MODEL="$(EMBEDDING_MODEL)" \
		$(DOCKER_IMAGE)

docker-compose-up: ## Start services with docker-compose
	@echo "🐳 Starting services with docker-compose..."
	@docker-compose up -d
	@echo "✅ Services started"

docker-compose-down: ## Stop docker-compose services
	@echo "🐳 Stopping docker-compose services..."
	@docker-compose down
	@echo "✅ Services stopped"

docker-compose-logs: ## View docker-compose logs
	@docker-compose logs -f

# Quality assurance targets
lint: ## Run linter
	@echo "🔍 Running linter..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "⚠️  golangci-lint not installed. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; \
		go vet ./...; \
		go fmt ./...; \
	fi
	@echo "✅ Linting complete"

format: ## Format Go code
	@echo "🎨 Formatting code..."
	@go fmt ./...
	@echo "✅ Code formatted"

vet: ## Run go vet
	@echo "🔍 Running go vet..."
	@go vet ./...
	@echo "✅ Vet complete"

security: ## Run security scanner (gosec)
	@echo "🔒 Running security scan..."
	@if command -v gosec >/dev/null 2>&1; then \
		gosec ./...; \
	else \
		echo "⚠️  gosec not installed. Install with: go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest"; \
	fi
	@echo "✅ Security scan complete"

# Utility targets
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@rm -f $(BINARY_NAME)
	@rm -f $(INDEXER_BINARY)

	@rm -f coverage.out coverage.html
	@docker image prune -f
	@echo "✅ Clean complete"

check-env: ## Verify environment variables
	@if [ -z "$(OPENAI_API_KEY)" ]; then \
		echo "❌ OPENAI_API_KEY environment variable is required"; \
		echo "   Set it with: export OPENAI_API_KEY='your-api-key'"; \
		echo "   Or create a .env file with: OPENAI_API_KEY=your-api-key"; \
		exit 1; \
	fi
	@echo "✅ Environment variables OK"

install-tools: ## Install development tools
	@echo "🛠️  Installing development tools..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
	@echo "✅ Development tools installed"

# API testing targets
test-api: check-index ## Test API endpoints (requires running server)
	@echo "🧪 Testing API endpoints..."
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8080/healthz || (echo "❌ Health check failed" && exit 1)
	@echo "✅ Health check passed"
	@echo "Testing query endpoint..."
	@curl -f -X POST http://localhost:8080/query \
		-H "Content-Type: application/json" \
		-d '{"query": "God is love", "k": 3}' \
		| jq . || (echo "❌ Query test failed" && exit 1)
	@echo "✅ API tests passed"

# Release targets
release-build: ## Build optimized release binaries
	@echo "🚀 Building release binaries..."
	@mkdir -p dist
	@GOOS=linux GOARCH=$(TARGET_ARCH) go build -ldflags="-s -w" -o dist/$(BINARY_NAME)-linux-amd64 main.go
	@GOOS=darwin GOARCH=$(TARGET_ARCH) go build -ldflags="-s -w" -o dist/$(BINARY_NAME)-darwin-amd64 main.go
	@GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o dist/$(BINARY_NAME)-darwin-arm64 main.go  # Keep arm64 for local, as TARGET_ARCH is for linux builds
	@GOOS=windows GOARCH=$(TARGET_ARCH) go build -ldflags="-s -w" -o dist/$(BINARY_NAME)-windows-amd64.exe main.go
	@echo "✅ Release binaries built in dist/"

# Full development workflow
dev-setup: deps install-tools build-indexer indexer ## Complete development setup
	@echo "🎉 Development environment ready!"
	@echo "   Next steps:"
	@echo "   1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'"
	@echo "   2. Run the server: make run"
	@echo "   3. Test the API: make test-api"

# Full quality check
qa: format vet lint test security ## Run all quality assurance checks
	@echo "✅ All quality checks passed!"

# Production deployment check
deploy-check: test check-index docker-build ## Pre-deployment verification
	@echo "✅ Ready for deployment!"
