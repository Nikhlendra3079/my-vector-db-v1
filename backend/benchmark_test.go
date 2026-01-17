package main

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func BenchmarkSearchSpeed(b *testing.B) {
	// 1. Setup Database
	store := NewVectorStore()
	dim := 768 // Standard dimension for nomic-embed-text
	numRecords := 10000

	fmt.Printf("Generating %d dummy vectors...\n", numRecords)
	for i := 0; i < numRecords; i++ {
		vec := make(Vector, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		store.AddItem(fmt.Sprintf("id-%d", i), vec, nil, "default")
	}

	// 2. Create a Query Vector
	query := make(Vector, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}

	// 3. Run Benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		results := store.Search(query, 5, "default", "", "")
		duration := time.Since(start)

		if i == 0 {
			fmt.Printf("Search took: %v for %d records\n", duration, numRecords)
			fmt.Printf("Top Result ID: %s, Score: %f\n", results[0].ID, results[0].Score)
		}
	}
}
