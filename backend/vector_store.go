package main

import (
	"container/heap"
	"encoding/json"
	"math"
	"os"
	"runtime"
	"sync"
)

type Vector []float32

// SearchResult used by the Min-Heap
type SearchResult struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"`
}

// ResultHeap implements heap.Interface for Top-K tracking
type ResultHeap []SearchResult

func (h ResultHeap) Len() int           { return len(h) }
func (h ResultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score } // Min-Heap
func (h ResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *ResultHeap) Push(x any)        { *h = append(*h, x.(SearchResult)) }
func (h *ResultHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type Record struct {
	ID        string            `json:"id"`
	Vector    Vector            `json:"vector"`
	Quantized []int8            `json:"quantized,omitempty"`
	Metadata  map[string]string `json:"metadata"`
	Namespace string            `json:"namespace"`
}

type VectorStore struct {
	sync.RWMutex
	Records []Record
	// O(1) Lookup for Metadata
	IDMap map[string]int
}

func NewVectorStore() *VectorStore {
	return &VectorStore{
		Records: []Record{},
		IDMap:   make(map[string]int),
	}
}

// DotProduct with loop unrolling to hint SIMD optimization
func DotProduct(a, b Vector) float32 {
	var sum float32
	n := len(a)
	// Manual unrolling for performance
	for i := 0; i < n-3; i += 4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
	}
	for i := (n / 4) * 4; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func Normalize(v Vector) Vector {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	mag := float32(math.Sqrt(float64(sum)))
	if mag == 0 {
		return v
	}
	res := make(Vector, len(v))
	for i := range v {
		res[i] = v[i] / mag
	}
	return res
}

// Scalar Quantization: Reduces memory footprint
func Quantize(v Vector) []int8 {
	res := make([]int8, len(v))
	for i, val := range v {
		res[i] = int8(val * 127.0)
	}
	return res
}

func (vs *VectorStore) AddItem(id string, vector Vector, meta map[string]string, namespace string) {
	vs.Lock()
	defer vs.Unlock()

	norm := Normalize(vector)
	record := Record{
		ID:        id,
		Vector:    norm,
		Quantized: Quantize(norm),
		Metadata:  meta,
		Namespace: namespace,
	}

	if idx, exists := vs.IDMap[id]; exists {
		vs.Records[idx] = record
	} else {
		vs.IDMap[id] = len(vs.Records)
		vs.Records = append(vs.Records, record)
	}
}

func (vs *VectorStore) Search(query Vector, k int, namespace string, filterKey, filterVal string) []SearchResult {
	vs.RLock()
	defer vs.RUnlock()

	q := Normalize(query)
	numWorkers := runtime.NumCPU()
	workChan := make(chan []SearchResult, numWorkers)
	var wg sync.WaitGroup

	chunkSize := (len(vs.Records) + numWorkers - 1) / numWorkers

	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		if start >= len(vs.Records) {
			break
		}
		end := start + chunkSize
		if end > len(vs.Records) {
			end = len(vs.Records)
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			h := &ResultHeap{}
			heap.Init(h)

			for j := s; j < e; j++ {
				rec := vs.Records[j]

				// Namespace & Pre-filtering
				if namespace != "" && rec.Namespace != namespace {
					continue
				}
				if filterKey != "" && rec.Metadata[filterKey] != filterVal {
					continue
				}

				score := DotProduct(q, rec.Vector)
				res := SearchResult{ID: rec.ID, Score: score}

				if h.Len() < k {
					heap.Push(h, res)
				} else if score > (*h)[0].Score {
					heap.Pop(h)
					heap.Push(h, res)
				}
			}

			results := make([]SearchResult, h.Len())
			for idx := h.Len() - 1; idx >= 0; idx-- {
				results[idx] = heap.Pop(h).(SearchResult)
			}
			workChan <- results
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(workChan)
	}()

	finalHeap := &ResultHeap{}
	heap.Init(finalHeap)
	for chunk := range workChan {
		for _, res := range chunk {
			if finalHeap.Len() < k {
				heap.Push(finalHeap, res)
			} else if res.Score > (*finalHeap)[0].Score {
				heap.Pop(finalHeap)
				heap.Push(finalHeap, res)
			}
		}
	}

	finalResults := make([]SearchResult, finalHeap.Len())
	for i := finalHeap.Len() - 1; i >= 0; i-- {
		finalResults[i] = heap.Pop(finalHeap).(SearchResult)
	}
	return finalResults
}

func (vs *VectorStore) Save(filename string) error {
	vs.RLock()
	defer vs.RUnlock()
	data, err := json.Marshal(vs.Records)
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (vs *VectorStore) Load(filename string) error {
	vs.Lock()
	defer vs.Unlock()
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	err = json.Unmarshal(data, &vs.Records)
	if err != nil {
		return err
	}

	vs.IDMap = make(map[string]int)
	for i, rec := range vs.Records {
		vs.IDMap[rec.ID] = i
	}
	return nil
}
