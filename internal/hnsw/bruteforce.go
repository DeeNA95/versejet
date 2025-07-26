package hnsw

/*
#cgo CFLAGS: -I${SRCDIR}/csrc
#cgo LDFLAGS: -L${SRCDIR}/csrc -lvector_search
#include <stdlib.h>
#include "vector_search.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// BruteForceSearch performs brute force k-NN cosine similarity search using the C implementation.
// Takes a slice of vectors, a query vector, number of neighbors k, and a similarity threshold.
// Returns a slice of matched indices or an error.
func BruteForceSearch(vectors []*Vector, query *Vector, k int, similarityThreshold float32) ([]int, error) {
	if len(vectors) == 0 {
		return nil, errors.New("input vectors slice is empty")
	}
	if query == nil || query.cvec == nil {
		return nil, errors.New("query vector is nil")
	}
	if k <= 0 {
		return nil, errors.New("k must be positive")
	}

	// Create a C array for the vectors
	cVectors := (*C.Vector)(C.malloc(C.size_t(len(vectors)) * C.size_t(unsafe.Sizeof(C.Vector{}))))
	if cVectors == nil {
		return nil, errors.New("failed to allocate memory for C vectors array")
	}
	defer C.free(unsafe.Pointer(cVectors))

	// Copy Go Vector pointers into the C array
	cVectorArray := (*[1 << 30]C.Vector)(unsafe.Pointer(cVectors))[:len(vectors):len(vectors)]
	for i, vec := range vectors {
		if vec == nil || vec.cvec == nil {
			return nil, errors.New("one of the input vectors is nil")
		}
		// Copy the C Vector struct contents (shallow copy is enough)
		cVectorArray[i] = *vec.cvec
	}

	var outCount C.int

	// Call the C brute force search function
	cResults := C.brute_force_knn_search(
		cVectors,
		C.int(len(vectors)),
		query.cvec,
		C.int(k),
		C.float(similarityThreshold),
		&outCount,
	)

	if cResults == nil || outCount == 0 {
		return nil, errors.New("no results returned from brute force search")
	}
	defer C.free(unsafe.Pointer(cResults))

	count := int(outCount)
	results := make([]int, count)
	cResultArray := (*[1 << 30]C.int)(unsafe.Pointer(cResults))[:count:count]
	for i := 0; i < count; i++ {
		results[i] = int(cResultArray[i])
	}

	return results, nil
}
