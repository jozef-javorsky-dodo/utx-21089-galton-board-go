package main

import (
	"bufio"
	"context"
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
	"os/signal"
	"runtime"
	"sync"
	"time"
)

type Config struct {
	NumBalls    uint64
	ImagePath   string
	NumWorkers  int
	Width       int
	Height      int
	NumLevels   int
	PegRadius   float64
	BallRadius  float64
	PegSpacingX float64
	PegSpacingY float64
	Gravity     float64
	Restitution float64
	Fuzz        float64
	Dt          float64
}

func DefaultConfig() Config {
	return Config{
		NumBalls:    1000000,
		ImagePath:   "galton_board.png",
		NumWorkers:  runtime.NumCPU(),
		Width:       1920,
		Height:      1080,
		NumLevels:   100,
		PegRadius:   2.0,
		BallRadius:  1.5,
		PegSpacingX: 18.0,
		PegSpacingY: 15.0,
		Gravity:     981.0,
		Restitution: 0.6,
		Fuzz:        10.0,
		Dt:          0.005,
	}
}

func (c *Config) Validate() error {
	var errs []error
	if c.Width <= 0 || c.Height <= 0 {
		errs = append(errs, errors.New("dimensions must be positive"))
	}
	if c.NumWorkers <= 0 {
		errs = append(errs, errors.New("workers must be positive"))
	}
	if c.NumBalls == 0 {
		errs = append(errs, errors.New("balls must be positive"))
	}
	if c.NumLevels <= 0 {
		errs = append(errs, errors.New("levels must be positive"))
	}
	if c.PegRadius <= 0 || c.BallRadius <= 0 {
		errs = append(errs, errors.New("radii must be positive"))
	}
	if c.PegSpacingX <= 0 || c.PegSpacingY <= 0 {
		errs = append(errs, errors.New("spacing must be positive"))
	}
	if c.Gravity <= 0 {
		errs = append(errs, errors.New("gravity must be positive"))
	}
	if c.Dt <= 0 {
		errs = append(errs, errors.New("dt must be positive"))
	}
	if c.ImagePath == "" {
		errs = append(errs, errors.New("path must not be empty"))
	}
	return errors.Join(errs...)
}

func ParseFlags() Config {
	cfg := DefaultConfig()
	flag.Uint64Var(&cfg.NumBalls, "balls", cfg.NumBalls, "")
	flag.StringVar(&cfg.ImagePath, "out", cfg.ImagePath, "")
	flag.IntVar(&cfg.NumWorkers, "workers", cfg.NumWorkers, "")
	flag.IntVar(&cfg.Width, "width", cfg.Width, "")
	flag.IntVar(&cfg.Height, "height", cfg.Height, "")
	flag.IntVar(&cfg.NumLevels, "levels", cfg.NumLevels, "")
	flag.Float64Var(&cfg.PegRadius, "peg_radius", cfg.PegRadius, "")
	flag.Float64Var(&cfg.BallRadius, "ball_radius", cfg.BallRadius, "")
	flag.Float64Var(&cfg.PegSpacingX, "peg_spacing_x", cfg.PegSpacingX, "")
	flag.Float64Var(&cfg.PegSpacingY, "peg_spacing_y", cfg.PegSpacingY, "")
	flag.Float64Var(&cfg.Gravity, "gravity", cfg.Gravity, "")
	flag.Float64Var(&cfg.Restitution, "restitution", cfg.Restitution, "")
	flag.Float64Var(&cfg.Fuzz, "fuzz", cfg.Fuzz, "")
	flag.Float64Var(&cfg.Dt, "dt", cfg.Dt, "")
	flag.Parse()
	return cfg
}

type Simulator interface {
	Simulate(ctx context.Context) ([]uint64, error)
}

type Renderer interface {
	Render(dist []uint64) *image.RGBA
}

type Exporter interface {
	Export(img *image.RGBA) error
}

type Board struct {
	cfg Config
}

func NewBoard(cfg Config) *Board {
	return &Board{cfg: cfg}
}

func (b *Board) Simulate(ctx context.Context) ([]uint64, error) {
	workerDists := make([][]uint64, b.cfg.NumWorkers)
	var wg sync.WaitGroup

	ballsPerWorker := b.cfg.NumBalls / uint64(b.cfg.NumWorkers)
	remainder := b.cfg.NumBalls % uint64(b.cfg.NumWorkers)

	seeds1 := make([]uint64, b.cfg.NumWorkers)
	seeds2 := make([]uint64, b.cfg.NumWorkers)
	for w := 0; w < b.cfg.NumWorkers; w++ {
		seed1, seed2, err := b.generateSeeds()
		if err != nil {
			return nil, fmt.Errorf("failed to generate seeds: %w", err)
		}
		seeds1[w] = seed1
		seeds2[w] = seed2
	}

	for w := 0; w < b.cfg.NumWorkers; w++ {
		workerDists[w] = make([]uint64, b.cfg.Width)
		balls := ballsPerWorker
		if uint64(w) < remainder {
			balls++
		}
		wg.Add(1)
		go b.simulateWorker(ctx, balls, seeds1[w], seeds2[w], workerDists[w], &wg)
	}

	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return b.aggregateDistributions(workerDists), nil
}

func (b *Board) generateSeeds() (uint64, uint64, error) {
	var buf [16]byte
	if _, err := crand.Read(buf[:]); err != nil {
		return 0, 0, err
	}
	return binary.LittleEndian.Uint64(buf[:8]), binary.LittleEndian.Uint64(buf[8:]), nil
}

func (b *Board) aggregateDistributions(workerDists [][]uint64) []uint64 {
	dist := make([]uint64, b.cfg.Width)
	for _, localDist := range workerDists {
		for i, count := range localDist {
			dist[i] += count
		}
	}
	return dist
}

func (b *Board) simulateWorker(ctx context.Context, ballsToSimulate uint64, seed1, seed2 uint64, localDist []uint64, wg *sync.WaitGroup) {
	defer wg.Done()
	rng := rand.New(rand.NewPCG(seed1, seed2))
	for i := uint64(0); i < ballsToSimulate; i++ {
		if i&1023 == 0 {
			select {
			case <-ctx.Done():
				return
			default:
			}
		}
		bin := b.simulateBall(rng)
		localDist[bin]++
	}
}

func (b *Board) simulateBall(rng *rand.Rand) int {
	x := float64(b.cfg.Width) / 2.0
	x += (rng.Float64() - 0.5) * 0.1
	y := 0.0
	vx, vy := 0.0, 0.0

	maxY := float64(b.cfg.NumLevels) * b.cfg.PegSpacingY
	minDist := b.cfg.PegRadius + b.cfg.BallRadius
	minDistSq := minDist * minDist
	halfDtSqGravity := 0.5 * b.cfg.Gravity * b.cfg.Dt * b.cfg.Dt

	for y < maxY {
		x += vx * b.cfg.Dt
		y += vy*b.cfg.Dt + halfDtSqGravity
		vy += b.cfg.Gravity * b.cfg.Dt

		if x < 0 {
			x = 0
			vx = -vx * b.cfg.Restitution
		} else if x > float64(b.cfg.Width-1) {
			x = float64(b.cfg.Width - 1)
			vx = -vx * b.cfg.Restitution
		}

		row := int(math.Round(y / b.cfg.PegSpacingY))
		if row >= 0 && row < b.cfg.NumLevels {
			pegY := float64(row) * b.cfg.PegSpacingY
			offset := 0.0
			if row&1 != 0 {
				offset = b.cfg.PegSpacingX / 2.0
			}

			col := int(math.Round((x - offset) / b.cfg.PegSpacingX))
			pegX := float64(col)*b.cfg.PegSpacingX + offset

			dx := x - pegX
			dy := y - pegY
			distSq := dx*dx + dy*dy

			if distSq < minDistSq && distSq > 0.0001 {
				dist := math.Sqrt(distSq)
				nx := dx / dist
				ny := dy / dist

				overlap := minDist - dist
				x += nx * overlap
				y += ny * overlap

				dot := vx*nx + vy*ny
				if dot < 0 {
					vx = (vx - 2*dot*nx) * b.cfg.Restitution
					vy = (vy - 2*dot*ny) * b.cfg.Restitution
					vx += (rng.Float64() - 0.5) * b.cfg.Fuzz
					vy += (rng.Float64() - 0.5) * b.cfg.Fuzz
				}
			}
		}
	}

	bin := int(math.Round(x))
	if bin < 0 {
		return 0
	}
	if bin > b.cfg.Width-1 {
		return b.cfg.Width - 1
	}
	return bin
}

type GraphRenderer struct {
	cfg Config
}

func NewGraphRenderer(cfg Config) *GraphRenderer {
	return &GraphRenderer{cfg: cfg}
}

func (r *GraphRenderer) Render(dist []uint64) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, r.cfg.Width, r.cfg.Height))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{R: 10, G: 10, B: 15, A: 255}}, image.Point{}, draw.Src)

	maxCount := r.maxDistCount(dist)
	if maxCount == 0 {
		return img
	}

	colorMap := r.generateColorMap()
	stride := img.Stride
	invMaxCount := 1.0 / float64(maxCount)
	fHeight := float64(r.cfg.Height)

	barHeights := make([]int, r.cfg.Width)
	for x := 0; x < r.cfg.Width && x < len(dist); x++ {
		bh := int(float64(dist[x]) * invMaxCount * fHeight)
		if bh > r.cfg.Height {
			bh = r.cfg.Height
		}
		barHeights[x] = bh
	}

	for py := 0; py < r.cfg.Height; py++ {
		y := r.cfg.Height - 1 - py
		c := colorMap[y]
		pyOffset := py * stride
		for x := 0; x < r.cfg.Width; x++ {
			if y < barHeights[x] {
				idx := pyOffset + x*4
				img.Pix[idx] = c.R
				img.Pix[idx+1] = c.G
				img.Pix[idx+2] = c.B
				img.Pix[idx+3] = c.A
			}
		}
	}
	return img
}

func (r *GraphRenderer) maxDistCount(dist []uint64) uint64 {
	var maxCount uint64
	for _, count := range dist {
		if count > maxCount {
			maxCount = count
		}
	}
	return maxCount
}

func (r *GraphRenderer) lerp(a, b, t float64) uint8 {
	return uint8(a + (b-a)*t)
}

func (r *GraphRenderer) generateColorMap() []color.RGBA {
	cmap := make([]color.RGBA, r.cfg.Height)
	invHeight := 1.0 / float64(r.cfg.Height)
	for y := 0; y < r.cfg.Height; y++ {
		t := float64(y) * invHeight
		var red, green, blue uint8
		if t < 0.5 {
			t2 := t * 2.0
			red = r.lerp(10, 220, t2)
			green = r.lerp(10, 20, t2)
			blue = r.lerp(40, 60, t2)
		} else {
			t2 := t*2.0 - 1.0
			red = r.lerp(220, 255, t2)
			green = r.lerp(20, 215, t2)
			blue = r.lerp(60, 0, t2)
		}
		cmap[y] = color.RGBA{R: red, G: green, B: blue, A: 255}
	}
	return cmap
}

type PNGExporter struct {
	path string
}

func NewPNGExporter(path string) *PNGExporter {
	return &PNGExporter{path: path}
}

func (e *PNGExporter) Export(img *image.RGBA) error {
	tmpPath := e.path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}

	defer func() {
		_ = f.Close()
		_ = os.Remove(tmpPath)
	}()

	bw := bufio.NewWriter(f)
	if err := png.Encode(bw, img); err != nil {
		return fmt.Errorf("encode png: %w", err)
	}
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("flush writer: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync file: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}

	if err := os.Rename(tmpPath, e.path); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}
	return nil
}

type Application struct {
	cfg      Config
	sim      Simulator
	renderer Renderer
	exporter Exporter
}

func NewApplication(cfg Config) *Application {
	return &Application{
		cfg:      cfg,
		sim:      NewBoard(cfg),
		renderer: NewGraphRenderer(cfg),
		exporter: NewPNGExporter(cfg.ImagePath),
	}
}

func (a *Application) Run(ctx context.Context) error {
	if err := a.cfg.Validate(); err != nil {
		return fmt.Errorf("configuration error: %w", err)
	}
	startTime := time.Now()
	distribution, err := a.sim.Simulate(ctx)
	if err != nil {
		return fmt.Errorf("simulation error: %w", err)
	}
	img := a.renderer.Render(distribution)
	if err := a.exporter.Export(img); err != nil {
		return fmt.Errorf("export error: %w", err)
	}
	fmt.Printf("Completed in %s. Saved to %s\n", time.Since(startTime), a.cfg.ImagePath)
	return nil
}

func main() {
	cfg := ParseFlags()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	app := NewApplication(cfg)
	if err := app.Run(ctx); err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "Simulation canceled")
			os.Exit(130)
		}
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
