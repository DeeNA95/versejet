package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"versejet/internal/index"

	"github.com/sashabaranov/go-openai"
)

// EmbeddingGenerator interface for generating embeddings
type EmbeddingGenerator interface {
	GenerateEmbedding(text string) ([]float32, error)
}

// OpenAIEmbeddingGenerator implements EmbeddingGenerator using OpenAI
type OpenAIEmbeddingGenerator struct {
	client *openai.Client
	model  string
}

// NewOpenAIEmbeddingGenerator creates a new OpenAI embedding generator
func NewOpenAIEmbeddingGenerator(apiKey, model string) *OpenAIEmbeddingGenerator {
	return &OpenAIEmbeddingGenerator{
		client: openai.NewClient(apiKey),
		model:  model,
	}
}

// GenerateEmbedding creates an embedding vector for the given text
func (g *OpenAIEmbeddingGenerator) GenerateEmbedding(text string) ([]float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.EmbeddingModel(g.model),
	}

	resp, err := g.client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai embedding request failed: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data received")
	}

	// Convert []float64 to []float32 for consistency
	embedding64 := resp.Data[0].Embedding
	embedding32 := make([]float32, len(embedding64))
	for i, val := range embedding64 {
		embedding32[i] = float32(val)
	}

	return embedding32, nil
}

// Handler manages API endpoints and dependencies
type Handler struct {
	verseIndex         *index.VerseIndex
	embeddingGenerator EmbeddingGenerator
	logger             *log.Logger
}

// NewHandler creates a new API handler with dependencies
func NewHandler(verseIndex *index.VerseIndex, openaiAPIKey, embeddingModel string, logger *log.Logger) *Handler {
	return &Handler{
		verseIndex:         verseIndex,
		embeddingGenerator: NewOpenAIEmbeddingGenerator(openaiAPIKey, embeddingModel),
		logger:             logger,
	}
}

// NewHandlerWithGenerator creates a new API handler with custom embedding generator
func NewHandlerWithGenerator(verseIndex *index.VerseIndex, generator EmbeddingGenerator, logger *log.Logger) *Handler {
	return &Handler{
		verseIndex:         verseIndex,
		embeddingGenerator: generator,
		logger:             logger,
	}
}

// QueryRequest represents the incoming search query
type QueryRequest struct {
	Query                   string   `json:"query"`
	K                       int      `json:"k,omitempty"`
	SearchWidth             *int     `json:"search_width,omitempty"`
	MaxDistanceComputations *int     `json:"max_distance_computations,omitempty"`
	AccuracyThreshold       *float32 `json:"accuracy_threshold,omitempty"`
	UseApproximateSearch    *bool    `json:"use_approximate_search,omitempty"`
}

// Query embedding helper (to avoid breaking existing API)
func (qr *QueryRequest) QueryEmbedding() []float32 {
	// Place holder: in your real code you should create embedding here or accept it from client.
	// For now just return empty slice
	return []float32{}
}

// QueryResponse represents the search results
type QueryResponse struct {
	Results []VerseResult `json:"results"`
	Query   string        `json:"query"`
	Count   int           `json:"count"`
}

// VerseResult represents a single verse result with context
type VerseResult struct {
	Ref      string  `json:"ref"`
	Text     string  `json:"text"`
	NextFive string  `json:"next_five"`
	Score    float32 `json:"score"`
}

// HandleQuery processes semantic search queries
func (h *Handler) HandleQuery(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	h.logger.Printf("🔍 Processing query request from %s", r.RemoteAddr)

	// Only allow POST requests
	if r.Method != http.MethodPost {
		h.sendError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse JSON request
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Printf("❌ Failed to decode request: %v", err)
		h.sendError(w, "Invalid JSON request", http.StatusBadRequest)
		return
	}

	// Validate query
	if req.Query == "" {
		h.sendError(w, "Query cannot be empty", http.StatusBadRequest)
		return
	}

	// Set default k value
	if req.K <= 0 {
		req.K = 20
	}
	if req.K > 50 {
		req.K = 50
	}

	h.logger.Printf("📝 Query: '%s', k=%d", req.Query, req.K)

	// Generate embedding for query
	h.logger.Println("🧠 Generating query embedding...")
	queryEmbedding, err := h.embeddingGenerator.GenerateEmbedding(req.Query)
	if err != nil {
		h.logger.Printf("❌ Failed to generate embedding: %v", err)
		h.sendError(w, "Failed to process query", http.StatusInternalServerError)
		return
	}

	embeddingTime := time.Since(startTime)
	h.logger.Printf("✅ Generated embedding in %v", embeddingTime)

	// Search for similar verses
	h.logger.Println("🔎 Searching for similar verses...")
	searchStart := time.Now()
	results, err := h.verseIndex.Search(queryEmbedding, req.K)
	if err != nil {
		h.logger.Printf("❌ Search failed: %v", err)
		h.sendError(w, "Search failed", http.StatusInternalServerError)
		return
	}

	searchTime := time.Since(searchStart)
	h.logger.Printf("✅ Found %d results in %v", len(results), searchTime)

	// Convert to response format
	verseResults := make([]VerseResult, len(results))
	for i, result := range results {
		verseResults[i] = VerseResult{
			Ref:      result.Verse.Ref,
			Text:     result.Verse.Text,
			NextFive: result.Verse.NextFive,
			Score:    result.Score,
		}
	}

	response := QueryResponse{
		Results: verseResults,
		Query:   req.Query,
		Count:   len(verseResults),
	}

	totalTime := time.Since(startTime)
	h.logger.Printf("🎯 Query completed in %v (embedding: %v, search: %v)", totalTime, embeddingTime, searchTime)

	// Send JSON response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

// sendError sends a JSON error response
func (h *Handler) sendError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResponse := map[string]interface{}{
		"error":     message,
		"status":    statusCode,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	json.NewEncoder(w).Encode(errorResponse)
}
