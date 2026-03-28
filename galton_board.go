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
		return errors.New("width and height must be positive")
	}
	if c.NumWorkers <= 0 {
		return errors.New("number of workers must be positive")
	}
	if c.NumBalls == 0 {
		return errors.New("number of balls must be non-zero")
	}
	if c.NumLevels <= 0 {
		return errors.New("number of levels must be positive")
	}
	if c.GravityConstant <= 0 {
		return errors.New("gravity constant must be positive")
	}
	if c.ImagePath == "" {
		return errors.New("image path cannot be empty")
	}
	if c.MinProb < 0 || c.MaxProb > 1 || c.MinProb >= c.MaxProb {
		return errors.New("probability clamps must be within [0, 1] and min < max")
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

type Simulator struct {
	config Config
}

func NewSimulator(cfg Config) *Simulator {
	return &Simulator{config: cfg}
}

func (s *Simulator) Run() ([]uint64, error) {
	workerDists := make([][]uint64, s.config.NumWorkers)
	var wg sync.WaitGroup

	ballsPerWorker := s.config.NumBalls / uint64(s.config.NumWorkers)
	remainder := s.config.NumBalls % uint64(s.config.NumWorkers)

	for w := 0; w < s.config.NumWorkers; w++ {
		workerDists[w] = make([]uint64, s.config.Width)

		seed1, seed2, err := s.generateSeeds()
		if err != nil {
			return nil, err
		}

		balls := ballsPerWorker
		if uint64(w) < remainder {
			balls++
		}

		wg.Add(1)
		go s.worker(balls, seed1, seed2, workerDists[w], &wg)
	}

	wg.Wait()
	return s.aggregateDistributions(workerDists), nil
}

func (s *Simulator) generateSeeds() (uint64, uint64, error) {
	var b [16]byte
	if _, err := crand.Read(b[:]); err != nil {
		return 0, 0, fmt.Errorf("failed to generate random seed: %w", err)
	}
	return binary.LittleEndian.Uint64(b[:8]), binary.LittleEndian.Uint64(b[8:]), nil
}

func (s *Simulator) aggregateDistributions(workerDists [][]uint64) []uint64 {
	dist := make([]uint64, s.config.Width)
	for _, localDist := range workerDists {
		for i, count := range localDist {
			dist[i] += count
		}
	}
	return dist
}

func (s *Simulator) worker(ballsToSimulate uint64, seed1, seed2 uint64, localDist []uint64, wg *sync.WaitGroup) {
	defer wg.Done()
	rng := rand.New(rand.NewPCG(seed1, seed2))

	dt := 1.0 / math.Sqrt(s.config.GravityConstant)
	baseFuzzStep := s.config.BaseFuzz * dt
	fuzzVarStep := s.config.FuzzVariance * dt

	startX := float64(s.config.Width) * 0.5
	maxBin := s.config.Width - 1

	for b := uint64(0); b < ballsToSimulate; b++ {
		bin := s.simulateBall(rng, startX, baseFuzzStep, fuzzVarStep)
		if bin < 0 {
			bin = 0
		} else if bin > maxBin {
			bin = maxBin
		}
		localDist[bin]++
	}
}

func (s *Simulator) simulateBall(rng *rand.Rand, startX, baseFuzzStep, fuzzVarStep float64) int {
	x := startX + (rng.Float64()-0.5)*2.0
	vx := 0.0

	for i := 0; i < s.config.NumLevels; i++ {
		probRight := 0.5 + vx*s.config.VelocityBias
		if probRight < s.config.MinProb {
			probRight = s.config.MinProb
		} else if probRight > s.config.MaxProb {
			probRight = s.config.MaxProb
		}

		fuzz := baseFuzzStep + rng.Float64()*fuzzVarStep

		if rng.Float64() < probRight {
			vx += fuzz
			x += s.config.StepSize
		} else {
			vx -= fuzz
			x -= s.config.StepSize
		}

		vx *= s.config.BounceDampening
		x += (rng.Float64() - 0.5) * s.config.PositionalFuzz
	}
	return int(math.Round(x))
}

type Renderer struct {
	config Config
}

func NewRenderer(cfg Config) *Renderer {
	return &Renderer{config: cfg}
}

func (r *Renderer) Render(dist []uint64) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, r.config.Width, r.config.Height))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{R: 10, G: 10, B: 15, A: 255}}, image.Point{}, draw.Src)

	maxCount := r.getMaxCount(dist)
	if maxCount == 0 {
		return img
	}

	colorMap := r.generateColorMap()
	stride := img.Stride
	invMaxCount := 1.0 / float64(maxCount)
	fHeight := float64(r.config.Height)

	for x, count := range dist {
		if count == 0 {
			continue
		}
		barHeight := int(float64(count) * invMaxCount * fHeight)

		for y := 0; y < barHeight && y < r.config.Height; y++ {
			c := colorMap[y]
			py := r.config.Height - 1 - y
			idx := py*stride + x*4

			img.Pix[idx] = c.R
			img.Pix[idx+1] = c.G
			img.Pix[idx+2] = c.B
			img.Pix[idx+3] = c.A
		}
	}
	return img
}

func (r *Renderer) getMaxCount(dist []uint64) uint64 {
	var maxCount uint64
	for _, count := range dist {
		if count > maxCount {
			maxCount = count
		}
	}
	return maxCount
}

func (r *Renderer) lerp(a, b, t float64) uint8 {
	return uint8(a + (b-a)*t)
}

func (r *Renderer) generateColorMap() []color.RGBA {
	cmap := make([]color.RGBA, r.config.Height)
	invHeight := 1.0 / float64(r.config.Height)

	for y := 0; y < r.config.Height; y++ {
		t := float64(y) * invHeight

		var red, green, blue uint8
		if t < 0.5 {
			t2 := t * 2.0
			red = r.lerp(10, 220, t2)
			green = r.lerp(10, 20, t2)
			blue = r.lerp(40, 60, t2)
		} else {
			t2 := (t - 0.5) * 2.0
			red = r.lerp(220, 255, t2)
			green = r.lerp(20, 215, t2)
			blue = r.lerp(60, 0, t2)
		}

		cmap[y] = color.RGBA{R: red, G: green, B: blue, A: 255}
	}
	return cmap
}

func (r *Renderer) Save(img *image.RGBA, path string) (err error) {
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
	if err := png.Encode(bw, img); err != nil {
		return fmt.Errorf("failed to encode png: %w", err)
	}

	if err := bw.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %w", err)
	}

	return nil
}

func runSimulation(cfg Config) error {
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	startTime := time.Now()

	simulator := NewSimulator(cfg)
	distribution, err := simulator.Run()
	if err != nil {
		return fmt.Errorf("simulation failed: %w", err)
	}

	simulationTime := time.Since(startTime)

	renderer := NewRenderer(cfg)
	img := renderer.Render(distribution)

	if err := renderer.Save(img, cfg.ImagePath); err != nil {
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
