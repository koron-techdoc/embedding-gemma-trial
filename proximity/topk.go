package main

import (
	"flag"
	"io"
	"log"
	"math"
	"os"
	"sort"

	"github.com/koron/embedding-gemma-demo/proximity/internal/matrix"
)

var (
	top           = 0
	out io.Writer = os.Stdout
)

func sortAndTopK(row *matrix.Row) {
	sort.SliceStable(row.Destinations, func(i, j int) bool {
		if math.IsNaN(row.Destinations[i].Distance) {
			return false
		}
		if math.IsNaN(row.Destinations[j].Distance) {
			return true
		}
		return row.Destinations[i].Distance < row.Destinations[j].Distance
	})
	if top > 0 {
		row.Destinations = row.Destinations[:top]
	}
}

func run(name string) error {
	m, err := matrix.LoadMatrix(name)
	if err != nil {
		return err
	}
	for i := range m.Rows {
		sortAndTopK(&m.Rows[i])
	}
	return matrix.Write(out, m)
}

func main() {
	flag.IntVar(&top, "top", 0, "show only cloesest top K (default: 0 = all)")
	flag.Parse()

	name := flag.Arg(0)
	err := run(name)
	if err != nil {
		log.Fatal(err)
	}
}
