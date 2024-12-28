package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

const (
	boardWidth  = 800
	boardHeight = 400
	numBalls    = 10000
	imagePath   = "galton_board.png"
)

type GaltonBoard struct {
	width, height, numBalls int
	distribution            []int
	random                  *rand.Rand
}

func NewGaltonBoard(width, height, numBalls int) *GaltonBoard {
	return &GaltonBoard{
		width:        width,
		height:       height,
		numBalls:     numBalls,
		distribution: make([]int, width),
		random:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (gb *GaltonBoard) Simulate() {
	for i := 0; i < gb.numBalls; i++ {
		gb.distribution[gb.calculateBin()]++
	}
}

func (gb *GaltonBoard) CreateImage() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, gb.width, gb.height))
	maxCount := gb.findMaxCount()
	for i, count := range gb.distribution {
		height := int(float64(count) / float64(maxCount) * float64(gb.height))
		for j := 0; j < height; j++ {
			img.Set(i, gb.height-j-1, color.RGBA{255, 0, 0, 255})
		}
	}
	return img
}

func (gb *GaltonBoard) calculateBin() int {
	bin := gb.width / 2
	for i := 0; i < gb.height; i++ {
		if gb.random.Intn(2) == 0 {
			bin--
		} else {
			bin++
		}
	}
	return int(math.Min(math.Max(float64(bin), 0), float64(gb.width-1)))
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

func saveImage(img *image.RGBA) {
	f, err := os.Create(imagePath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		log.Fatal(err)
	}
}

func runSimulation() {
	gb := NewGaltonBoard(boardWidth, boardHeight, numBalls)
	gb.Simulate()
	img := gb.CreateImage()
	saveImage(img)
}

func main() {
	runSimulation()
	fmt.Println(" ••• GALTON BOARD SIMUL COMPLETED ••• IMAGE ••• ", imagePath)
}
