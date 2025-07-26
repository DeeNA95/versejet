package hnsw

/*
#cgo CFLAGS: -I${SRCDIR}/csrc
#cgo LDFLAGS: -L${SRCDIR}/csrc -lvector_search
#include <stdlib.h>
#include "vector_search.h"

// Helper function to create Vector from Go float array
Vector* create_vector_from_array(float* data, int len) {
    Vector* vec = (Vector*)malloc(sizeof(Vector));
    vec->len = len;
    vec->data = (float*)malloc(sizeof(float) * len);
    for (int i = 0; i < len; i++) {
        vec->data[i] = data[i];
    }
    return vec;
}

// Helper function to free Vector
void free_vector(Vector* vec) {
    if (vec) {
        if (vec->data) {
            free(vec->data);
        }
        free(vec);
    }
}

// Serialization/deserialization wrappers

// Serialize the HNSWGraph struct to a byte buffer allocated by C
int serialize_hnsw_graph(HNSWGraph* graph, char** out_buffer, int* out_size);

// Deserialize the byte buffer into a new HNSWGraph pointer
HNSWGraph* deserialize_hnsw_graph(const char* buffer, int size);

// Free a buffer allocated by the C serialization function
void free_serialized_buffer(char* buffer);

// Wrapper for brute force cosine similarity k-NN search
int* brute_force_knn_search_wrapper(Vector* vectors, int len, Vector* query, int k, float similarity_threshold, int* out_count) {
    return brute_force_knn_search(vectors, len, query, k, similarity_threshold, out_count);
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Vector represents a vector backed by C Vector struct
type Vector struct {
	cvec *C.Vector
}

// NewVector creates a new Vector from Go float32 slice
func NewVector(data []float32) *Vector {
	if len(data) == 0 {
		return nil
	}
	cArray := (*C.float)(unsafe.Pointer(&data[0]))
	cvec := C.create_vector_from_array(cArray, C.int(len(data)))
	return &Vector{cvec: cvec}
}

// Free releases the underlying C vector memory
func (v *Vector) Free() {
	if v.cvec != nil {
		C.free_vector(v.cvec)
		v.cvec = nil
	}
}

// HNSWGraph wraps the C HNSWGraph struct
type HNSWGraph struct {
	graph *C.HNSWGraph
}

// BuildHNSWGraph builds an HNSW graph for the given vectors with parameters
func BuildHNSWGraph(vectors [][]float32, maxConnections int, maxConnectionsLayerZero int, levelGenerationFactor float32) (*HNSWGraph, error) {
	vectorCount := len(vectors)
	if vectorCount == 0 {
		return nil, errors.New("no vectors provided")
	}

	// Allocate C array for vectors
	cVectors := (*C.Vector)(C.malloc(C.size_t(vectorCount) * C.size_t(unsafe.Sizeof(C.Vector{}))))
	defer C.free(unsafe.Pointer(cVectors))

	// Convert Go vectors to C vectors
	cVectorArray := (*[1 << 30]C.Vector)(unsafe.Pointer(cVectors))[:vectorCount:vectorCount]
	for i, vec := range vectors {
		if len(vec) == 0 {
			continue
		}
		cVectorArray[i].len = C.int(len(vec))
		cVectorArray[i].data = (*C.float)(C.malloc(C.size_t(len(vec)) * C.size_t(unsafe.Sizeof(C.float(0)))))
		cDataArray := (*[1 << 30]C.float)(unsafe.Pointer(cVectorArray[i].data))[:len(vec):len(vec)]
		for j, val := range vec {
			cDataArray[j] = C.float(val)
		}
	}

	// Call C function
	cGraph := C.build_hnsw_graph(cVectors,
		C.int(vectorCount),
		C.int(maxConnections),
		C.int(maxConnectionsLayerZero),
		C.float(levelGenerationFactor),
		C.int(maxConnections*2)) // construction_search_width = maxConnections * 2 as default

	if cGraph == nil {
		return nil, errors.New("failed to build HNSW graph")
	}

	return &HNSWGraph{graph: cGraph}, nil
}

// Free releases memory allocated for the HNSWGraph
func (g *HNSWGraph) Free() {
	if g.graph != nil {
		C.free_hnsw_graph(g.graph)
		g.graph = nil
	}
}

// SearchConfig holds options for search
type SearchConfig struct {
	SearchWidth             int
	MaxDistanceComputations int
	AccuracyThreshold       float32
	UseApproximateSearch    bool
}

// SearchKNN performs k-nearest neighbor search on HNSW graph given a query vector and config
func (g *HNSWGraph) SearchKNN(query []float32, k int, config *SearchConfig) ([]int, error) {
	if g.graph == nil {
		return nil, errors.New("HNSW graph is nil")
	}
	if len(query) == 0 {
		return nil, errors.New("query vector is empty")
	}

	// Create C vector index wrapper
	cIndex := (*C.VectorIndex)(C.malloc(C.size_t(unsafe.Sizeof(C.VectorIndex{}))))
	defer C.free(unsafe.Pointer(cIndex))
	cIndex.vectors = g.graph.original_vectors
	cIndex.len = g.graph.node_count
	cIndex.hnsw_graph = g.graph
	cIndex.use_hnsw_optimization = 1

	cQueryVector := NewVector(query)
	defer cQueryVector.Free()

	var cConfig *C.SearchConfig

	// Build C SearchConfig struct or nil to use defaults
	if config != nil {
		cConfig = (*C.SearchConfig)(C.malloc(C.size_t(unsafe.Sizeof(C.SearchConfig{}))))
		defer C.free(unsafe.Pointer(cConfig))
		cConfig.search_width = C.int(config.SearchWidth)
		cConfig.max_distance_computations = C.int(config.MaxDistanceComputations)
		cConfig.accuracy_threshold = C.float(config.AccuracyThreshold)
		if config.UseApproximateSearch {
			cConfig.use_approximate_search = 1
		} else {
			cConfig.use_approximate_search = 0
		}
	} else {
		cConfig = nil
	}

	cResults := C.hnsw_knn_search(cIndex, cQueryVector.cvec, C.int(k), cConfig)
	if cResults == nil {
		return nil, errors.New("hnsw_knn_search returned nil results")
	}
	defer C.free(unsafe.Pointer(cResults))

	// Convert C int* array to Go slice
	// We do not have result count from C directly, so assume k elements
	results := make([]int, k)
	cArray := (*[1 << 30]C.int)(unsafe.Pointer(cResults))[:k:k]
	for i := 0; i < k; i++ {
		results[i] = int(cArray[i])
	}

	return results, nil
}
