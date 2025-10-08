package main

import (
	"flag"
	"fmt"
	"log"
	"math"

	"github.com/koron/embedding-gemma-demo/proximity/internal/matrix"
)

func find(dests []matrix.Destination, name string) bool {
	for _, d := range dests {
		if d.Name == name {
			return true
		}
	}
	return false
}

func accrary(base, target *matrix.Matrix) float64 {
	if len(base.Rows) != len(target.Rows) {
		return math.NaN()
	}
	var hit, total float64
	for i := range len(target.Rows) {
		r0 := &base.Rows[i]
		r1 := &target.Rows[i]
		if len(r0.Destinations) != len(r1.Destinations) {
			return math.NaN()
		}
		for _, d := range r1.Destinations {
			if find(r0.Destinations, d.Name) {
				hit++
			}
			total++
		}
	}
	return hit / total
}

func run(base, target string) error {
	mat0, err := matrix.LoadFile(base)
	if err != nil {
		return err
	}
	mat1, err := matrix.LoadFile(target)
	if err != nil {
		return err
	}
	a := accrary(mat0, mat1)
	fmt.Printf("Top-K accurary: %f\n", a)
	return nil
}

func main() {
	flag.Parse()
	if flag.NArg() < 2 {
		log.Fatal("too few arguments, at least 2")
	}
	err := run(flag.Arg(0), flag.Arg(1))
	if err != nil {
		log.Fatal(err)
	}
}
