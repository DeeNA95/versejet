package index

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestNewVerseIndex(t *testing.T) {
	verseIndex := NewVerseIndex()

	if verseIndex == nil {
		t.Fatal("NewVerseIndex() returned nil")
	}

	if len(verseIndex.Verses) != 0 {
		t.Errorf("Expected empty verse slice, got %d verses", len(verseIndex.Verses))
	}
}

func TestAddVerse(t *testing.T) {
	verseIndex := NewVerseIndex()

	verse := Verse{
		ID:        "GEN.1.1",
		Ref:       "Genesis 1:1",
		Text:      "In the beginning God created the heaven and the earth.",
		NextFive:  "And the earth was without form...",
		Embedding: []float32{0.1, 0.2, 0.3},
	}

	verseIndex.AddVerse(verse)

	if len(verseIndex.Verses) != 1 {
		t.Errorf("Expected 1 verse, got %d", len(verseIndex.Verses))
	}

	if verseIndex.Verses[0].ID != "GEN.1.1" {
		t.Errorf("Expected verse ID 'GEN.1.1', got '%s'", verseIndex.Verses[0].ID)
	}
}

func TestGetVerseCount(t *testing.T) {
	verseIndex := NewVerseIndex()

	if verseIndex.GetVerseCount() != 0 {
		t.Errorf("Expected count 0, got %d", verseIndex.GetVerseCount())
	}

	verse := Verse{ID: "GEN.1.1", Embedding: []float32{0.1, 0.2}}
	verseIndex.AddVerse(verse)

	if verseIndex.GetVerseCount() != 1 {
		t.Errorf("Expected count 1, got %d", verseIndex.GetVerseCount())
	}
}

func TestGetByRef(t *testing.T) {
	verseIndex := NewVerseIndex()

	verse := Verse{
		ID:        "GEN.1.1",
		Ref:       "Genesis 1:1",
		Text:      "In the beginning God created the heaven and the earth.",
		Embedding: []float32{0.1, 0.2, 0.3},
	}

	verseIndex.AddVerse(verse)

	// Test existing verse
	foundVerse, exists := verseIndex.GetByRef("Genesis 1:1")
	if !exists {
		t.Error("Expected to find verse by reference")
	}
	if foundVerse.ID != "GEN.1.1" {
		t.Errorf("Expected ID 'GEN.1.1', got '%s'", foundVerse.ID)
	}

	// Test non-existing verse
	_, exists = verseIndex.GetByRef("Non-existent verse")
	if exists {
		t.Error("Expected not to find non-existent verse")
	}
}

func TestGetByID(t *testing.T) {
	verseIndex := NewVerseIndex()

	verse := Verse{
		ID:        "GEN.1.1",
		Ref:       "Genesis 1:1",
		Text:      "In the beginning God created the heaven and the earth.",
		Embedding: []float32{0.1, 0.2, 0.3},
	}

	verseIndex.AddVerse(verse)

	// Test existing verse
	foundVerse, exists := verseIndex.GetByID("GEN.1.1")
	if !exists {
		t.Error("Expected to find verse by ID")
	}
	if foundVerse.Ref != "Genesis 1:1" {
		t.Errorf("Expected ref 'Genesis 1:1', got '%s'", foundVerse.Ref)
	}

	// Test non-existing verse
	_, exists = verseIndex.GetByID("NON.1.1")
	if exists {
		t.Error("Expected not to find non-existent verse")
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
		delta    float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1.0, 0.0, 0.0},
			b:        []float32{1.0, 0.0, 0.0},
			expected: 1.0,
			delta:    0.001,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1.0, 0.0},
			b:        []float32{0.0, 1.0},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1.0, 0.0},
			b:        []float32{-1.0, 0.0},
			expected: -1.0,
			delta:    0.001,
		},
		{
			name:     "different lengths",
			a:        []float32{1.0, 0.0},
			b:        []float32{1.0, 0.0, 0.0},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "zero vector",
			a:        []float32{0.0, 0.0},
			b:        []float32{1.0, 1.0},
			expected: 0.0,
			delta:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > float64(tt.delta) {
				t.Errorf("Expected %f, got %f", tt.expected, result)
			}
		})
	}
}

func TestSearch(t *testing.T) {
	verseIndex := NewVerseIndex()

	// Add test verses
	verses := []Verse{
		{
			ID:        "GEN.1.1",
			Ref:       "Genesis 1:1",
			Text:      "In the beginning God created the heaven and the earth.",
			Embedding: []float32{1.0, 0.0, 0.0},
		},
		{
			ID:        "JOH.3.16",
			Ref:       "John 3:16",
			Text:      "For God so loved the world...",
			Embedding: []float32{0.8, 0.6, 0.0}, // Similar to query
		},
		{
			ID:        "PSA.23.1",
			Ref:       "Psalms 23:1",
			Text:      "The LORD is my shepherd...",
			Embedding: []float32{0.0, 1.0, 0.0}, // Different from query
		},
	}

	for _, verse := range verses {
		verseIndex.AddVerse(verse)
	}

	// Test search with query similar to John 3:16
	// Search for similar verses
	query := []float32{0.7, 0.7, 0.1}
	results, err := verseIndex.Search(query, 2)

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Results should be sorted by similarity (descending)
	if len(results) >= 2 && results[0].Score < results[1].Score {
		t.Error("Results should be sorted by similarity score (descending)")
	}

	// Test with empty query
	_, err = verseIndex.Search([]float32{}, 5)
	if err == nil {
		t.Error("Expected error for empty query embedding")
	}

	// Test with k=0 (should default to 20)
	results, err = verseIndex.Search(query, 0)
	if err != nil {
		t.Fatalf("Search with k=0 failed: %v", err)
	}
	if len(results) != 3 { // We only have 3 verses
		t.Errorf("Expected all 3 verses, got %d", len(results))
	}

	// Test with k > 50 (should cap at 50)
	results, err = verseIndex.Search(query, 100)
	if err != nil {
		t.Fatalf("Search with k=100 failed: %v", err)
	}
	if len(results) != 3 { // We only have 3 verses
		t.Errorf("Expected all 3 verses, got %d", len(results))
	}
}

func TestSaveAndLoadGob(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := ioutil.TempDir("", "versejet_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	gobFile := filepath.Join(tempDir, "test-index.gob")

	// Create verse index with test data
	originalIndex := NewVerseIndex()

	verses := []Verse{
		{
			ID:        "GEN.1.1",
			Ref:       "Genesis 1:1",
			Text:      "In the beginning God created the heaven and the earth.",
			NextFive:  "And the earth was without form...",
			Embedding: []float32{0.1, 0.2, 0.3, 0.4},
		},
		{
			ID:        "JOH.3.16",
			Ref:       "John 3:16",
			Text:      "For God so loved the world...",
			NextFive:  "That whosoever believeth...",
			Embedding: []float32{0.5, 0.6, 0.7, 0.8},
		},
	}

	for _, verse := range verses {
		originalIndex.AddVerse(verse)
	}

	// Save to gob file
	err = originalIndex.SaveToGob(gobFile)
	if err != nil {
		t.Fatalf("Failed to save gob file: %v", err)
	}

	// Load from gob file
	loadedIndex, err := LoadFromGob(gobFile)
	if err != nil {
		t.Fatalf("Failed to load gob file: %v", err)
	}

	// Verify loaded data matches original
	if len(loadedIndex.Verses) != len(originalIndex.Verses) {
		t.Errorf("Expected %d verses, got %d", len(originalIndex.Verses), len(loadedIndex.Verses))
	}

	for i, originalVerse := range originalIndex.Verses {
		loadedVerse := loadedIndex.Verses[i]

		if originalVerse.ID != loadedVerse.ID {
			t.Errorf("Verse %d ID mismatch: expected '%s', got '%s'", i, originalVerse.ID, loadedVerse.ID)
		}

		if originalVerse.Ref != loadedVerse.Ref {
			t.Errorf("Verse %d Ref mismatch: expected '%s', got '%s'", i, originalVerse.Ref, loadedVerse.Ref)
		}

		if originalVerse.Text != loadedVerse.Text {
			t.Errorf("Verse %d Text mismatch: expected '%s', got '%s'", i, originalVerse.Text, loadedVerse.Text)
		}

		if !reflect.DeepEqual(originalVerse.Embedding, loadedVerse.Embedding) {
			t.Errorf("Verse %d Embedding mismatch", i)
		}
	}

	// Test loading non-existent file
	_, err = LoadFromGob("non-existent-file.gob")
	if err == nil {
		t.Error("Expected error when loading non-existent file")
	}
}

func TestSearchResultSorting(t *testing.T) {
	verseIndex := NewVerseIndex()

	// Add verses with known similarity scores to a query
	// Using vectors that will have predictable cosine similarity with query
	verses := []Verse{
		{
			ID:        "LOW",
			Ref:       "Low Similarity",
			Text:      "Low similarity verse",
			Embedding: []float32{0.0, 0.0, 1.0}, // Orthogonal to query, low similarity
		},
		{
			ID:        "HIGH",
			Ref:       "High Similarity",
			Text:      "High similarity verse",
			Embedding: []float32{1.0, 0.0, 0.0}, // Aligned with query, high similarity
		},
		{
			ID:        "MED",
			Ref:       "Medium Similarity",
			Text:      "Medium similarity verse",
			Embedding: []float32{0.5, 0.5, 0.0}, // Partially aligned, medium similarity
		},
	}

	for _, verse := range verses {
		verseIndex.AddVerse(verse)
	}

	// Query that should be most similar to HIGH embedding (aligned with x-axis)
	query := []float32{1.0, 0.0, 0.0}
	results, err := verseIndex.Search(query, 3)

	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Verify sorting: HIGH should be first, then MED, then LOW
	if results[0].Verse.ID != "HIGH" {
		t.Errorf("Expected first result to be HIGH, got %s", results[0].Verse.ID)
	}

	if results[1].Verse.ID != "MED" {
		t.Errorf("Expected second result to be MED, got %s", results[1].Verse.ID)
	}

	if results[2].Verse.ID != "LOW" {
		t.Errorf("Expected third result to be LOW, got %s", results[2].Verse.ID)
	}

	// Verify scores are in descending order
	for i := 1; i < len(results); i++ {
		if results[i-1].Score < results[i].Score {
			t.Errorf("Results not sorted properly: score[%d]=%.3f < score[%d]=%.3f",
				i-1, results[i-1].Score, i, results[i].Score)
		}
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	// Benchmark with 1536-dimensional vectors (typical embedding size)
	dim := 1536
	a := make([]float32, dim)
	vecB := make([]float32, dim)

	for i := 0; i < dim; i++ {
		a[i] = float32(i) / float32(dim)
		vecB[i] = float32(dim-i) / float32(dim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cosineSimilarity(a, vecB)
	}
}

func BenchmarkSearch(b *testing.B) {
	verseIndex := NewVerseIndex()

	// Create a large index similar to the real one
	numVerses := 31000
	embeddingDim := 1536

	for i := 0; i < numVerses; i++ {
		embedding := make([]float32, embeddingDim)
		for j := 0; j < embeddingDim; j++ {
			embedding[j] = float32(i+j) / float32(embeddingDim)
		}

		verse := Verse{
			ID:        fmt.Sprintf("BENCH.%d.1", i),
			Ref:       fmt.Sprintf("Benchmark %d:1", i),
			Text:      fmt.Sprintf("Benchmark verse text %d", i),
			Embedding: embedding,
		}
		verseIndex.AddVerse(verse)
	}

	// Create query embedding
	query := make([]float32, embeddingDim)
	for i := 0; i < embeddingDim; i++ {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := verseIndex.Search(query, 20)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
