package index

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"

	"versejet/internal/hnsw"
)

// Verse represents a single Bible verse with its embedding and context
type Verse struct {
	ID        string    `json:"id"`        // e.g. "GEN.1.1"
	Ref       string    `json:"ref"`       // e.g. "Genesis 1:1"
	Text      string    `json:"text"`      // original punctuation-preserved text
	NextFive  string    `json:"next_five"` // concatenated text of next 5 verses
	Embedding []float32 `json:"embedding"` // 1536-dimensional embedding vector
}

// VerseIndex holds all verses and provides search functionality
type VerseIndex struct {
	Verses    []Verse `json:"verses"`
	hnswIndex *hnsw.HNSWGraph
}

// NewVerseIndex creates a new empty verse index
func NewVerseIndex() *VerseIndex {
	return &VerseIndex{
		Verses:    make([]Verse, 0),
		hnswIndex: nil,
	}
}

// AddVerse adds a verse to the index
func (vi *VerseIndex) AddVerse(verse Verse) {
	vi.Verses = append(vi.Verses, verse)
}

// SearchResult represents a search result with similarity score
type SearchResult struct {
	Verse Verse   `json:"verse"`
	Score float32 `json:"score"`
}

// Search performs cosine similarity search against all verses
func (vi *VerseIndex) Search(queryEmbedding []float32, k int) ([]SearchResult, error) {
	if len(queryEmbedding) == 0 {
		return nil, fmt.Errorf("query embedding cannot be empty")
	}
	if k <= 0 {
		k = 20
	}
	if k > 50 {
		k = 50
	}

	cQueryVec := hnsw.NewVector(queryEmbedding)
	if cQueryVec == nil {
		return nil, fmt.Errorf("failed to create query vector")
	}
	defer cQueryVec.Free()

	// Prepare vectors slice for C function
	vectors := make([]*hnsw.Vector, len(vi.Verses))
	for i, verse := range vi.Verses {
		vectors[i] = hnsw.NewVector(verse.Embedding)
		defer vectors[i].Free()
	}

	threshold := float32(0.5)

	ids, err := hnsw.BruteForceSearch(vectors, cQueryVec, k, threshold)
	if err != nil || len(ids) == 0 {
		// fallback to slow Go brute force if C function fails
		var results []SearchResult
		for _, verse := range vi.Verses {
			if len(verse.Embedding) != len(queryEmbedding) {
				continue
			}
			similarity := cosineSimilarity(queryEmbedding, verse.Embedding)
			if similarity >= threshold {
				results = append(results, SearchResult{
					Verse: verse,
					Score: similarity,
				})
			}
		}
		// Sort by descending score
		for i := 0; i < len(results)-1; i++ {
			for j := i + 1; j < len(results); j++ {
				if results[i].Score < results[j].Score {
					results[i], results[j] = results[j], results[i]
				}
			}
		}

		if k > len(results) {
			k = len(results)
		}

		return results[:k], nil
	}

	results := make([]SearchResult, 0, len(ids))
	for _, id := range ids {
		if id < 0 || id >= len(vi.Verses) {
			continue
		}
		verse := vi.Verses[id]
		similarity := cosineSimilarity(queryEmbedding, verse.Embedding)
		if similarity < threshold {
			continue
		}
		results = append(results, SearchResult{
			Verse: verse,
			Score: similarity,
		})
	}

	return results, nil
}

// calculateEuclideanDistance calculates Euclidean distance between two vectors
func calculateEuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// SaveHNSW writes a checkpoint of the built HNSW graph to disk.
/*
func (vi *VerseIndex) SaveHNSW(path string) error {
	if vi.hnswIndex == nil {
		return fmt.Errorf("HNSW index is nil, cannot save")
	}
	data, err := vi.hnswIndex.Serialize()
	if err != nil {
		return fmt.Errorf("failed to serialize HNSW graph: %w", err)
	}
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("gob encoding failed: %w", err)
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint file: %w", err)
	}
	defer file.Close()
	_, err = file.Write(buf.Bytes())
	if err != nil {
		return fmt.Errorf("failed to write checkpoint file: %w", err)
	}
	return nil
}
*/

// LoadHNSW loads a checkpointed HNSW graph from disk into memory.
/*
func (vi *VerseIndex) LoadHNSW(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open checkpoint file: %w", err)
	}
	defer file.Close()
	fileInfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat checkpoint file: %w", err)
	}
	dataBytes := make([]byte, fileInfo.Size())
	if _, err := file.Read(dataBytes); err != nil {
		return fmt.Errorf("failed to read checkpoint file: %w", err)
	}
	var decodedData []byte
	decoder := gob.NewDecoder(bytes.NewReader(dataBytes))
	if err := decoder.Decode(&decodedData); err != nil {
		return fmt.Errorf("gob decoding failed: %w", err)
	}
	graph, err := hnsw.Deserialize(decodedData)
	if err != nil {
		return fmt.Errorf("failed to deserialize HNSW graph: %w", err)
	}
	vi.hnswIndex = graph
	return nil
}
*/

// BuildHNSWIndex builds HNSW index from verse embeddings in the index
func (vi *VerseIndex) BuildHNSWIndex(maxConnections, maxConnectionsLayerZero int, levelFactor float32) error {
	vectors := make([][]float32, len(vi.Verses))
	for i, verse := range vi.Verses {
		vectors[i] = verse.Embedding
	}
	graph, err := hnsw.BuildHNSWGraph(vectors, maxConnections, maxConnectionsLayerZero, levelFactor)
	if err != nil {
		return err
	}
	vi.hnswIndex = graph
	return nil
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float32

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// SaveToGob serializes the verse index to a gob file
func (vi *VerseIndex) SaveToGob(filepath string) error {
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create gob file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(vi); err != nil {
		return fmt.Errorf("failed to encode verse index: %w", err)
	}

	return nil
}

// LoadFromGob deserializes a verse index from a gob file
func LoadFromGob(filepath string) (*VerseIndex, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open gob file: %w", err)
	}
	defer file.Close()

	var verseIndex VerseIndex
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&verseIndex); err != nil {
		return nil, fmt.Errorf("failed to decode verse index: %w", err)
	}

	return &verseIndex, nil
}

// GetVerseCount returns the number of verses in the index
func (vi *VerseIndex) GetVerseCount() int {
	return len(vi.Verses)
}

// GetByRef finds a verse by its reference string
func (vi *VerseIndex) GetByRef(ref string) (*Verse, bool) {
	for _, verse := range vi.Verses {
		if verse.Ref == ref {
			return &verse, true
		}
	}
	return nil, false
}

// GetByID finds a verse by its ID
func (vi *VerseIndex) GetByID(id string) (*Verse, bool) {
	for _, verse := range vi.Verses {
		if verse.ID == id {
			return &verse, true
		}
	}
	return nil, false
}
