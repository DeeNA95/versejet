package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"versejet/internal/index"
)

// MockEmbeddingGenerator implements EmbeddingGenerator for testing
type MockEmbeddingGenerator struct {
	shouldFail      bool
	customEmbedding func(string) ([]float32, error)
}

func (m *MockEmbeddingGenerator) GenerateEmbedding(text string) ([]float32, error) {
	if m.shouldFail {
		return nil, fmt.Errorf("mock embedding generation failed")
	}

	if m.customEmbedding != nil {
		return m.customEmbedding(text)
	}

	// Default mock implementation
	embedding := make([]float32, 1536)
	for i := range embedding {
		embedding[i] = 0.5 // Simple default embedding
	}
	return embedding, nil
}

// createMockHandler creates a handler with mock dependencies for testing
func createMockHandler(shouldFailOpenAI bool) *Handler {
	// Create test verse index
	verseIndex := index.NewVerseIndex()

	testVerses := []index.Verse{
		{
			ID:        "GEN.1.1",
			Ref:       "Genesis 1:1",
			Text:      "In the beginning God created the heaven and the earth.",
			NextFive:  "And the earth was without form, and void...",
			Embedding: []float32{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		{
			ID:        "JOH.3.16",
			Ref:       "John 3:16",
			Text:      "For God so loved the world, that he gave his only begotten Son...",
			NextFive:  "That whosoever believeth in him should not perish...",
			Embedding: []float32{0.6, 0.7, 0.8, 0.9, 1.0},
		},
		{
			ID:        "PSA.23.1",
			Ref:       "Psalms 23:1",
			Text:      "The LORD is my shepherd; I shall not want.",
			NextFive:  "He maketh me to lie down in green pastures...",
			Embedding: []float32{0.2, 0.4, 0.6, 0.8, 1.0},
		},
	}

	for _, verse := range testVerses {
		verseIndex.AddVerse(verse)
	}

	// Create mock embedding generator
	mockGenerator := &MockEmbeddingGenerator{
		shouldFail: shouldFailOpenAI,
	}

	// Create handler with mock generator
	handler := NewHandlerWithGenerator(
		verseIndex,
		mockGenerator,
		log.New(os.Stdout, "[TEST] ", 0),
	)

	return handler
}

func TestHandleQuery_Success(t *testing.T) {
	handler := createMockHandler(false)

	// Override embedding generation for testing
	mockGen := handler.embeddingGenerator.(*MockEmbeddingGenerator)
	mockGen.customEmbedding = func(text string) ([]float32, error) {
		// Return a mock embedding similar to one of our test verses
		return []float32{0.6, 0.7, 0.8, 0.9, 1.0}, nil
	}

	requestBody := QueryRequest{
		Query: "God so loved the world",
		K:     2,
	}
	body, _ := json.Marshal(requestBody)

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response QueryResponse
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if response.Query != "God so loved the world" {
		t.Errorf("Expected query 'God so loved the world', got '%s'", response.Query)
	}

	if len(response.Results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(response.Results))
	}

	if response.Count != 2 {
		t.Errorf("Expected count 2, got %d", response.Count)
	}

	// Verify result structure
	if len(response.Results) > 0 {
		result := response.Results[0]
		if result.Ref == "" {
			t.Error("Expected non-empty Ref")
		}
		if result.Text == "" {
			t.Error("Expected non-empty Text")
		}
		if result.Score == 0 {
			t.Error("Expected non-zero Score")
		}
	}
}

func TestHandleQuery_MethodNotAllowed(t *testing.T) {
	handler := createMockHandler(false)

	req := httptest.NewRequest(http.MethodGet, "/query", nil)
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405, got %d", w.Code)
	}
}

func TestHandleQuery_InvalidJSON(t *testing.T) {
	handler := createMockHandler(false)

	req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader("invalid json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestHandleQuery_EmptyQuery(t *testing.T) {
	handler := createMockHandler(false)

	requestBody := QueryRequest{
		Query: "",
		K:     10,
	}
	body, _ := json.Marshal(requestBody)

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestHandleQuery_DefaultK(t *testing.T) {
	handler := createMockHandler(false)

	// Override embedding generation for testing
	mockGen := handler.embeddingGenerator.(*MockEmbeddingGenerator)
	mockGen.customEmbedding = func(text string) ([]float32, error) {
		return []float32{0.1, 0.2, 0.3, 0.4, 0.5}, nil
	}

	requestBody := QueryRequest{
		Query: "test query",
		// K not specified, should default to 20
	}
	body, _ := json.Marshal(requestBody)

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response QueryResponse
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Should return all 3 verses since we only have 3 in the test index
	if len(response.Results) != 3 {
		t.Errorf("Expected 3 results (all verses), got %d", len(response.Results))
	}
}

func TestHandleQuery_KCapping(t *testing.T) {
	handler := createMockHandler(false)

	// Override embedding generation for testing
	mockGen := handler.embeddingGenerator.(*MockEmbeddingGenerator)
	mockGen.customEmbedding = func(text string) ([]float32, error) {
		return []float32{0.1, 0.2, 0.3, 0.4, 0.5}, nil
	}

	requestBody := QueryRequest{
		Query: "test query",
		K:     100, // Should be capped at 50
	}
	body, _ := json.Marshal(requestBody)

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response QueryResponse
	err := json.NewDecoder(w.Body).Decode(&response)
	if err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Should return all 3 verses since we only have 3 in the test index
	if len(response.Results) != 3 {
		t.Errorf("Expected 3 results (all verses), got %d", len(response.Results))
	}
}

func TestHandleQuery_EmbeddingFailure(t *testing.T) {
	handler := createMockHandler(false)

	// Set mock to fail
	mockGen := handler.embeddingGenerator.(*MockEmbeddingGenerator)
	mockGen.shouldFail = true

	requestBody := QueryRequest{
		Query: "test query",
		K:     10,
	}
	body, _ := json.Marshal(requestBody)

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.HandleQuery(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("Expected status 500, got %d", w.Code)
	}
}

func TestGenerateEmbedding_Success(t *testing.T) {
	handler := createMockHandler(false)

	// Test the mocked embedding generation
	embedding, err := handler.embeddingGenerator.GenerateEmbedding("test text")

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(embedding) != 1536 {
		t.Errorf("Expected embedding length 1536, got %d", len(embedding))
	}

	// Check that all values are 0.5 (default mock value)
	for i, val := range embedding {
		if val != 0.5 {
			t.Errorf("Expected embedding[%d] to be 0.5, got %f", i, val)
			break
		}
	}
}

func TestSendError(t *testing.T) {
	handler := createMockHandler(false)

	w := httptest.NewRecorder()
	handler.sendError(w, "Test error message", http.StatusBadRequest)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}

	var errorResponse map[string]interface{}
	err := json.NewDecoder(w.Body).Decode(&errorResponse)
	if err != nil {
		t.Fatalf("Failed to decode error response: %v", err)
	}

	if errorResponse["error"] != "Test error message" {
		t.Errorf("Expected error message 'Test error message', got '%v'", errorResponse["error"])
	}

	if errorResponse["status"] != float64(400) {
		t.Errorf("Expected status 400, got %v", errorResponse["status"])
	}

	if _, exists := errorResponse["timestamp"]; !exists {
		t.Error("Expected timestamp in error response")
	}
}

func TestResponseStructures(t *testing.T) {
	// Test QueryRequest JSON marshaling/unmarshaling
	req := QueryRequest{
		Query: "test query",
		K:     15,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal QueryRequest: %v", err)
	}

	var unmarshaled QueryRequest
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal QueryRequest: %v", err)
	}

	if unmarshaled.Query != req.Query {
		t.Errorf("Query mismatch: expected '%s', got '%s'", req.Query, unmarshaled.Query)
	}

	if unmarshaled.K != req.K {
		t.Errorf("K mismatch: expected %d, got %d", req.K, unmarshaled.K)
	}

	// Test QueryResponse JSON marshaling
	response := QueryResponse{
		Results: []VerseResult{
			{
				Ref:      "Genesis 1:1",
				Text:     "In the beginning...",
				NextFive: "And the earth was...",
				Score:    0.95,
			},
		},
		Query: "test query",
		Count: 1,
	}

	data, err = json.Marshal(response)
	if err != nil {
		t.Fatalf("Failed to marshal QueryResponse: %v", err)
	}

	var unmarshaledResponse QueryResponse
	err = json.Unmarshal(data, &unmarshaledResponse)
	if err != nil {
		t.Fatalf("Failed to unmarshal QueryResponse: %v", err)
	}

	if len(unmarshaledResponse.Results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(unmarshaledResponse.Results))
	}

	if unmarshaledResponse.Results[0].Ref != "Genesis 1:1" {
		t.Errorf("Ref mismatch: expected 'Genesis 1:1', got '%s'", unmarshaledResponse.Results[0].Ref)
	}
}

func BenchmarkHandleQuery(b *testing.B) {
	handler := createMockHandler(false)

	// Override embedding generation for consistent benchmarking
	mockGen := handler.embeddingGenerator.(*MockEmbeddingGenerator)
	mockGen.customEmbedding = func(text string) ([]float32, error) {
		return []float32{0.1, 0.2, 0.3, 0.4, 0.5}, nil
	}

	requestBody := QueryRequest{
		Query: "benchmark query",
		K:     20,
	}
	body, _ := json.Marshal(requestBody)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler.HandleQuery(w, req)

		if w.Code != http.StatusOK {
			b.Fatalf("Request failed with status %d", w.Code)
		}
	}
}
