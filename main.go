package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"versejet/internal/api"
	"versejet/internal/index"
)

func main() {
	// Load configuration
	config := loadConfig()

	// Initialize logger
	logger := log.New(os.Stdout, "[VERSEJET] ", log.LstdFlags|log.Lshortfile)
	logger.Println("🚀 Starting VerseJet server...")

	// Load verse index from gob file
	logger.Println("📚 Loading verse index...")
	verseIndex, err := index.LoadFromGob(config.IndexPath)
	if err != nil {
		logger.Fatalf("❌ Failed to load verse index: %v", err)
	}
	logger.Printf("✅ Loaded %d verses from index", len(verseIndex.Verses))

	// HNSW checkpoint startup logic
	checkpointPath := config.HNSWCheckpointPath
	// ensure checkpoint dir
	if dir := filepath.Dir(checkpointPath); dir != "" {
		if err := os.MkdirAll(dir, 0755); err != nil {
			logger.Printf("⚠️ Failed to create checkpoint dir %s: %v", dir, err)
		}
	}
	// Temporarily disable HNSW checkpoint loading and saving since not used with brute force
	/*
		if _, err := os.Stat(checkpointPath); err == nil {
			logger.Printf("📁 Found HNSW checkpoint at %s, loading...", checkpointPath)
			if err := verseIndex.LoadHNSW(checkpointPath); err != nil {
				logger.Printf("⚠️ Failed to load checkpoint: %v. Rebuilding...", err)
				if berr := verseIndex.BuildHNSWIndex(16, 32, 1.0/2.71828); berr != nil {
					logger.Printf("⚠️ Failed to build HNSW index: %v. Using brute-force.", berr)
				} else {
					logger.Println("✅ HNSW rebuilt successfully, saving checkpoint...")
					if serr := verseIndex.SaveHNSW(checkpointPath); serr != nil {
						logger.Printf("⚠️ Failed to save checkpoint: %v", serr)
					}
				}
			} else {
				logger.Println("✅ Loaded HNSW index from checkpoint")
			}
		} else {
			logger.Printf("📁 No checkpoint found at %s, building HNSW index...", checkpointPath)
			if berr := verseIndex.BuildHNSWIndex(16, 32, 1.0/2.71828); berr != nil {
				logger.Printf("⚠️ Failed to build HNSW index: %v. Using brute-force.", berr)
			} else {
				logger.Println("✅ HNSW index built, saving checkpoint...")
				if serr := verseIndex.SaveHNSW(checkpointPath); serr != nil {
					logger.Printf("⚠️ Failed to save checkpoint: %v", serr)
				} else {
					logger.Println("✅ HNSW checkpoint saved")
				}
			}
		}
	*/

	// Initialize API handler
	apiHandler := api.NewHandler(verseIndex, config.OpenAIAPIKey, config.EmbeddingModel, logger)

	// Setup HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		http.ServeFile(w, r, "index.html")
	})
	mux.HandleFunc("/query", apiHandler.HandleQuery)
	mux.HandleFunc("/healthz", handleHealth)

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", config.Port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.Printf("🌐 Server listening on port %d", config.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("❌ Server failed to start: %v", err)
		}
	}()

	// Wait for interrupt signal for graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Println("🛑 Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Printf("⚠️  Server forced to shutdown: %v", err)
	} else {
		logger.Println("✅ Server shutdown complete")
	}
}

type Config struct {
	Port               int    `json:"port"`
	IndexPath          string `json:"index_path"`
	OpenAIAPIKey       string `json:"openai_api_key"`
	EmbeddingModel     string `json:"embedding_model"`
	HNSWCheckpointPath string `json:"hnsw_checkpoint_path"`
}

// loadConfig loads configuration from environment variables with defaults
func loadConfig() *Config {
	config := &Config{
		Port:               getEnvInt("PORT", 8080),
		IndexPath:          getEnv("INDEX_PATH", "data/bible-index.gob"),
		OpenAIAPIKey:       getEnv("OPENAI_API_KEY", ""),
		EmbeddingModel:     getEnv("EMBEDDING_MODEL", "text-embedding-3-small"),
		HNSWCheckpointPath: getEnv("HNSW_CHECKPOINT_PATH", "data/hnsw_checkpoint.gob"),
	}

	if config.OpenAIAPIKey == "" {
		log.Fatal("❌ OPENAI_API_KEY environment variable is required")
	}

	return config
}

// getEnv gets environment variable with default value
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvInt gets environment variable as integer with default value
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

// handleHealth provides a health check endpoint
func handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	response := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"service":   "versejet",
	}

	json.NewEncoder(w).Encode(response)
}
