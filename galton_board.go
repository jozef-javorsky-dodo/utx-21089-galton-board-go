package main

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	mrand "math/rand"
	"os"
	"time"
)

const (
	boardWidth  = 800
	boardHeight = 400
	numBalls    = 100000
	imagePath   = "galton_board.png"
	numWorkers  = 16
)

type GaltonBoard struct {
	width, height, numBalls int
	distribution            []int
}

func NewGaltonBoard(width, height, numBalls int) *GaltonBoard {
	return &GaltonBoard{
		width:        width,
		height:       height,
		numBalls:     numBalls,
		distribution: make([]int, width),
	}
}

func (gb *GaltonBoard) worker(jobs <-chan int, results chan<- int, seed int64) {
	localRand := mrand.New(mrand.NewSource(seed))
	for range jobs {
		bin := gb.width / 2
		for i := 0; i < gb.height; i++ {
			if localRand.Intn(2) == 0 {
				bin--
			} else {
				bin++
			}
		}
		finalBin := int(math.Min(math.Max(float64(bin), 0), float64(gb.width-1)))
		results <- finalBin
	}
}

func (gb *GaltonBoard) Simulate() {
	jobs := make(chan int, gb.numBalls)
	results := make(chan int, gb.numBalls)

	for w := 0; w < numWorkers; w++ {
		var seed int64
		err := binary.Read(rand.Reader, binary.LittleEndian, &seed)
		if err != nil {
			log.Printf("could not generate a cryptographic seed, falling back to time-based seed: %v", err)
			seed = time.Now().UnixNano() + int64(w)
		}
		go gb.worker(jobs, results, seed)
	}

	for j := 0; j < gb.numBalls; j++ {
		jobs <- j
	}
	close(jobs)

	for a := 0; a < gb.numBalls; a++ {
		bin := <-results
		gb.distribution[bin]++
	}
	close(results)
}

func (gb *GaltonBoard) CreateImage() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, gb.width, gb.height))
	maxCount := gb.findMaxCount()
	if maxCount == 0 {
		return img
	}

	for i, count := range gb.distribution {
		heightRatio := float64(count) / float64(maxCount)
		barHeight := int(heightRatio * float64(gb.height))
		for j := 0; j < barHeight; j++ {
			img.Set(i, gb.height-1-j, color.RGBA{R: 220, G: 20, B: 60, A: 255})
		}
	}
	return img
}

func (gb *GaltonBoard) findMaxCount() int {
	maxCount := 0
	for _, count := range gb.distribution {
		if count > maxCount {
			maxCount = count
		}
	}
	return maxCount
}

func saveImage(img *image.RGBA, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create image file: %w", err)
	}
	defer f.Close()

	if err := png.Encode(f, img); err != nil {
		return fmt.Errorf("failed to encode image to png: %w", err)
	}
	return nil
}

func runSimulation() error {
	board := NewGaltonBoard(boardWidth, boardHeight, numBalls)

	startTime := time.Now()
	board.Simulate()
	simulationTime := time.Since(startTime)

	img := board.CreateImage()

	if err := saveImage(img, imagePath); err != nil {
		return err
	}

	fmt.Printf("Galton Board simulation completed in %s. Image saved to %s\n", simulationTime, imagePath)
	return nil
}

func main() {
	if err := runSimulation(); err != nil {
		log.Fatalf("An error occurred during the simulation: %v", err)
	}
}
