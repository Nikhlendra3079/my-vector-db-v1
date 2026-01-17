package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
)

var db *VectorStore

type AddRequest struct {
	ID        string            `json:"id"`
	Text      string            `json:"text"`
	Namespace string            `json:"namespace"`
	Metadata  map[string]string `json:"metadata"`
}

type QueryRequest struct {
	Text      string `json:"text"`
	K         int    `json:"k"`
	Namespace string `json:"namespace"`
	FilterKey string `json:"filter_key"`
	FilterVal string `json:"filter_val"`
}

func getEmbedding(text string) ([]float32, error) {
	reqBody := map[string]string{"model": "nomic-embed-text", "prompt": text}
	jsonData, _ := json.Marshal(reqBody)
	resp, err := http.Post("http://localhost:11434/api/embeddings", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var res struct {
		Embedding []float32 `json:"embedding"`
	}
	json.NewDecoder(resp.Body).Decode(&res)
	return res.Embedding, nil
}

func main() {
	db = NewVectorStore()
	db.Load("vectors.json")

	r := gin.Default()

	r.POST("/add", func(c *gin.Context) {
		var req AddRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		vec, err := getEmbedding(req.Text)
		if err != nil {
			c.JSON(500, gin.H{"error": "Embedding error"})
			return
		}

		if req.Metadata == nil {
			req.Metadata = make(map[string]string)
		}
		req.Metadata["text"] = req.Text

		db.AddItem(req.ID, Vector(vec), req.Metadata, req.Namespace)
		c.JSON(200, gin.H{"status": "success", "total": len(db.Records)})
	})

	r.POST("/query", func(c *gin.Context) {
		var req QueryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		if req.K == 0 {
			req.K = 5
		}

		queryVec, _ := getEmbedding(req.Text)
		results := db.Search(Vector(queryVec), req.K, req.Namespace, req.FilterKey, req.FilterVal)

		// O(1) Metadata Retrieval
		type DetailedResult struct {
			SearchResult
			Metadata map[string]string `json:"metadata"`
		}

		db.RLock()
		finalResponse := make([]DetailedResult, len(results))
		for i, res := range results {
			idx := db.IDMap[res.ID]
			finalResponse[i] = DetailedResult{
				SearchResult: res,
				Metadata:     db.Records[idx].Metadata,
			}
		}
		db.RUnlock()

		c.JSON(200, gin.H{"results": finalResponse})
	})

	srv := &http.Server{Addr: ":8080", Handler: r}
	go func() { srv.ListenAndServe() }()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	db.Save("vectors.json")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
}
