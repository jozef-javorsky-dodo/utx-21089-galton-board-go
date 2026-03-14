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
	"os"
	"time"
)

const (
	imageWidth      = 1920
	imageHeight     = 1080
	imagePath       = "galton_board.png"
	numWorkers      = 16
	numBalls        = 2000000
	numLevels       = 400
	stepSize        = 13.0
	gravityConstant = 9.81
	bounceDampening = 0.75
	fuzziness       = 0.25
)

type xorshift64 struct {
	state uint64
}

func (r *xorshift64) next() uint64 {
	x := r.state
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	r.state = x
	return x
}

func (r *xorshift64) float64() float64 {
	return float64(r.next()>>11) * (1.0 / 9007199254740992.0)
}

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

func (gb *GaltonBoard) worker(ballsToSimulate int, seed uint64, results chan<- []int) {
	rng := &xorshift64{state: seed}
	if rng.state == 0 {
		rng.state = 1
	}

	localDist := make([]int, gb.width)
	timeStep := 1.0 / math.Sqrt(gravityConstant)

	for b := 0; b < ballsToSimulate; b++ {
		x := float64(gb.width) / 2.0
		x += (rng.float64() - 0.5) * 2.0
		vx := 0.0

		for i := 0; i < numLevels; i++ {
			probRight := 0.5 + (vx * 0.15)
			if probRight < 0.05 {
				probRight = 0.05
			} else if probRight > 0.95 {
				probRight = 0.95
			}

			if rng.float64() < probRight {
				vx += (0.5 + rng.float64()*fuzziness) * timeStep
				x += stepSize
			} else {
				vx -= (0.5 + rng.float64()*fuzziness) * timeStep
				x -= stepSize
			}

			vx *= bounceDampening
			x += (rng.float64() - 0.5) * fuzziness
		}

		bin := int(math.Round(x))
		if bin < 0 {
			bin = 0
		} else if bin >= gb.width {
			bin = gb.width - 1
		}

		localDist[bin]++
	}
	results <- localDist
}

func (gb *GaltonBoard) Simulate() {
	results := make(chan []int, numWorkers)

	ballsPerWorker := gb.numBalls / numWorkers
	remainder := gb.numBalls % numWorkers

	for w := 0; w < numWorkers; w++ {
		var seed uint64
		err := binary.Read(rand.Reader, binary.LittleEndian, &seed)
		if err != nil {
			seed = uint64(time.Now().UnixNano()) + uint64(w)
		}

		balls := ballsPerWorker
		if w == 0 {
			balls += remainder
		}

		go gb.worker(balls, seed, results)
	}

	for w := 0; w < numWorkers; w++ {
		localDist := <-results
		for i, count := range localDist {
			gb.distribution[i] += count
		}
	}
	close(results)
}

func getGradientColor(y, maxH int) color.RGBA {
	if maxH == 0 {
		return color.RGBA{R: 10, G: 10, B: 40, A: 255}
	}

	t := float64(y) / float64(maxH)
	var r, g, b uint8

	if t < 0.5 {
		t2 := t * 2.0
		r = uint8(float64(10)*(1-t2) + float64(220)*t2)
		g = uint8(float64(10)*(1-t2) + float64(20)*t2)
		b = uint8(float64(40)*(1-t2) + float64(60)*t2)
	} else {
		t2 := (t - 0.5) * 2.0
		r = uint8(float64(220)*(1-t2) + float64(255)*t2)
		g = uint8(float64(20)*(1-t2) + float64(215)*t2)
		b = uint8(float64(60)*(1-t2) + float64(0)*t2)
	}

	return color.RGBA{R: r, G: g, B: b, A: 255}
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
			c := getGradientColor(j, gb.height)
			img.SetRGBA(i, gb.height-1-j, c)
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
	board := NewGaltonBoard(imageWidth, imageHeight, numBalls)

	startTime := time.Now()
	board.Simulate()
	simulationTime := time.Since(startTime)

	img := board.CreateImage()

	if err := saveImage(img, imagePath); err != nil {
		return err
	}

	fmt.Printf("Galton Board simulation of %d balls completed in %s. Image saved to %s\n", numBalls, simulationTime, imagePath)
	return nil
}

func main() {
	if err := runSimulation(); err != nil {
		log.Fatalf("An error occurred during the simulation: %v", err)
	}
}
