package main

import (
	"bufio"
	crand "crypto/rand"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"sync"
	"time"
)

type Config struct {
	Width           int
	Height          int
	ImagePath       string
	NumWorkers      int
	NumBalls        uint64
	NumLevels       int
	StepSize        float64
	GravityConstant float64
	BounceDampening float64
	VelocityBias    float64
	MinProb         float64
	MaxProb         float64
	BaseFuzz        float64
	FuzzVariance    float64
	PositionalFuzz  float64
}

func DefaultConfig() Config {
	return Config{
		Width:           1920,
		Height:          1080,
		ImagePath:       "galton_board.png",
		NumWorkers:      runtime.NumCPU(),
		NumBalls:        2000000,
		NumLevels:       400,
		StepSize:        13.0,
		GravityConstant: 9.81,
		BounceDampening: 0.75,
		VelocityBias:    0.15,
		MinProb:         0.05,
		MaxProb:         0.95,
		BaseFuzz:        0.5,
		FuzzVariance:    0.25,
		PositionalFuzz:  0.25,
	}
}

func (c *Config) Validate() error {
	if c.Width <= 0 || c.Height <= 0 {
		return errors.New("width and height must be greater than zero")
	}
	if c.NumWorkers <= 0 {
		return errors.New("number of workers must be greater than zero")
	}
	if c.NumBalls == 0 {
		return errors.New("number of balls must be greater than zero")
	}
	if c.NumLevels <= 0 {
		return errors.New("number of levels must be greater than zero")
	}
	if c.GravityConstant <= 0 {
		return errors.New("gravity constant must be greater than zero")
	}
	if c.ImagePath == "" {
		return errors.New("image path cannot be empty")
	}
	return nil
}

func parseFlags() Config {
	cfg := DefaultConfig()
	flag.IntVar(&cfg.Width, "width", cfg.Width, "")
	flag.IntVar(&cfg.Height, "height", cfg.Height, "")
	flag.StringVar(&cfg.ImagePath, "out", cfg.ImagePath, "")
	flag.IntVar(&cfg.NumWorkers, "workers", cfg.NumWorkers, "")
	flag.Uint64Var(&cfg.NumBalls, "balls", cfg.NumBalls, "")
	flag.IntVar(&cfg.NumLevels, "levels", cfg.NumLevels, "")
	flag.Float64Var(&cfg.StepSize, "step", cfg.StepSize, "")
	flag.Parse()
	return cfg
}

func lerp(a, b, t float64) uint8 {
	return uint8(a + (b-a)*t)
}

func worker(cfg Config, ballsToSimulate uint64, seed1, seed2 uint64, localDist []uint64, wg *sync.WaitGroup) {
	defer wg.Done()

	rng := rand.New(rand.NewPCG(seed1, seed2))
	dt := 1.0 / math.Sqrt(cfg.GravityConstant)
	baseFuzzStep := cfg.BaseFuzz * dt
	fuzzVarStep := cfg.FuzzVariance * dt
	startX := float64(cfg.Width) * 0.5
	maxBin := cfg.Width - 1

	for b := uint64(0); b < ballsToSimulate; b++ {
		x := startX + (rng.Float64()-0.5)*2.0
		vx := 0.0

		for i := 0; i < cfg.NumLevels; i++ {
			probRight := 0.5 + vx*cfg.VelocityBias
			if probRight < cfg.MinProb {
				probRight = cfg.MinProb
			} else if probRight > cfg.MaxProb {
				probRight = cfg.MaxProb
			}

			fuzz := baseFuzzStep + rng.Float64()*fuzzVarStep

			if rng.Float64() < probRight {
				vx += fuzz
				x += cfg.StepSize
			} else {
				vx -= fuzz
				x -= cfg.StepSize
			}

			vx *= cfg.BounceDampening
			x += (rng.Float64() - 0.5) * cfg.PositionalFuzz
		}

		bin := int(x + 0.5)
		if bin < 0 {
			bin = 0
		} else if bin > maxBin {
			bin = maxBin
		}

		localDist[bin]++
	}
}

func simulateGaltonBoard(cfg Config) ([]uint64, error) {
	workerDists := make([][]uint64, cfg.NumWorkers)
	var wg sync.WaitGroup

	ballsPerWorker := cfg.NumBalls / uint64(cfg.NumWorkers)
	remainder := cfg.NumBalls % uint64(cfg.NumWorkers)

	for w := 0; w < cfg.NumWorkers; w++ {
		workerDists[w] = make([]uint64, cfg.Width)
		var b [16]byte
		if _, err := crand.Read(b[:]); err != nil {
			return nil, fmt.Errorf("failed to generate random seed: %w", err)
		}
		s1 := binary.LittleEndian.Uint64(b[:8])
		s2 := binary.LittleEndian.Uint64(b[8:])

		balls := ballsPerWorker
		if w == 0 {
			balls += remainder
		}

		wg.Add(1)
		go worker(cfg, balls, s1, s2, workerDists[w], &wg)
	}

	wg.Wait()

	dist := make([]uint64, cfg.Width)
	for _, localDist := range workerDists {
		for i, count := range localDist {
			dist[i] += count
		}
	}
	return dist, nil
}

func generateColorMap(height int) []color.RGBA {
	cmap := make([]color.RGBA, height)
	invHeight := 1.0 / float64(height)
	for y := 0; y < height; y++ {
		t := float64(y) * invHeight
		var r, g, b uint8

		if t < 0.5 {
			t2 := t * 2.0
			r = lerp(10, 220, t2)
			g = lerp(10, 20, t2)
			b = lerp(40, 60, t2)
		} else {
			t2 := (t - 0.5) * 2.0
			r = lerp(220, 255, t2)
			g = lerp(20, 215, t2)
			b = lerp(60, 0, t2)
		}
		cmap[y] = color.RGBA{R: r, G: g, B: b, A: 255}
	}
	return cmap
}

func createDistributionImage(cfg Config, dist []uint64) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{R: 10, G: 10, B: 15, A: 255}}, image.Point{}, draw.Src)

	var maxCount uint64
	for _, count := range dist {
		if count > maxCount {
			maxCount = count
		}
	}

	if maxCount == 0 {
		return img
	}

	colorMap := generateColorMap(cfg.Height)
	stride := img.Stride
	invMaxCount := 1.0 / float64(maxCount)
	fHeight := float64(cfg.Height)

	for x, count := range dist {
		if count == 0 {
			continue
		}
		barHeight := int(float64(count) * invMaxCount * fHeight)

		for y := 0; y < barHeight; y++ {
			c := colorMap[y]
			py := cfg.Height - 1 - y
			idx := py*stride + x*4

			img.Pix[idx] = c.R
			img.Pix[idx+1] = c.G
			img.Pix[idx+2] = c.B
			img.Pix[idx+3] = c.A
		}
	}
	return img
}

func saveImage(img *image.RGBA, path string) (err error) {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove existing image: %w", err)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create image file: %w", err)
	}

	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("failed to close image file: %w", cerr)
		}
	}()

	bw := bufio.NewWriter(f)
	if err = png.Encode(bw, img); err != nil {
		return fmt.Errorf("failed to encode png: %w", err)
	}

	if err = bw.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %w", err)
	}

	return nil
}

func runSimulation(cfg Config) error {
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	startTime := time.Now()
	distribution, err := simulateGaltonBoard(cfg)
	if err != nil {
		return fmt.Errorf("simulation failed: %w", err)
	}
	simulationTime := time.Since(startTime)

	img := createDistributionImage(cfg, distribution)

	if err := saveImage(img, cfg.ImagePath); err != nil {
		return err
	}

	fmt.Printf("Galton Board simulation of %d balls completed in %s. Image saved to %s\n", cfg.NumBalls, simulationTime, cfg.ImagePath)
	return nil
}

func main() {
	cfg := parseFlags()
	if err := runSimulation(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}