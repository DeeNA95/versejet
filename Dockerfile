# Multi-stage Dockerfile for VerseJet

# Stage 1: Build stage using Golang with necessary tools
FROM golang:1.24-bookworm AS builder

# Install build dependencies for C library
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy go.mod and go.sum first for dependency caching
COPY go.mod go.sum ./

# Download Go dependencies
RUN go mod download

# Copy the rest of the source code
COPY . .

# Build the C library
RUN make build-c

# Build the indexer
RUN make build-indexer

# Generate the bible-index.gob using the indexer
RUN ./indexer -text verses-1769.json -embeddings VersejetKJV_recreated.json -output data/bible-index.gob

# Build the main application binary with CGO
RUN make build

# Stage 2: Runtime stage using lightweight Debian
FROM debian:bookworm-slim

# Install runtime dependencies if needed (minimal for CGO)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the main binary from builder
COPY --from=builder /app/versejet /app/versejet

# Copy the generated index file
COPY --from=builder /app/data/bible-index.gob /app/data/bible-index.gob

# Copy the index.html for serving
COPY --from=builder /app/index.html /app/index.html

# Expose the port the app runs on
EXPOSE 8080

# Set the entrypoint to the binary
# Environment variables like OPENAI_API_KEY should be set at runtime
CMD ["/app/versejet"]
