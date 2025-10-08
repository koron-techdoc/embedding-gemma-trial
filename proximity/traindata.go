package main

import (
	"bufio"
	"flag"
	"io"
	"log"
	"math"
	"os"
	"sort"

	"github.com/koron/embedding-gemma-demo/proximity/internal/matrix"
)

var (
	k             = 10
	out io.Writer = os.Stdout
)

func sortDestinations(row matrix.Row) {
	sort.SliceStable(row.Destinations, func(i, j int) bool {
		if math.IsNaN(row.Destinations[i].Distance) {
			return false
		}
		if math.IsNaN(row.Destinations[j].Distance) {
			return true
		}
		return row.Destinations[i].Distance < row.Destinations[j].Distance
	})
}

func run(name string) error {
	m, err := matrix.LoadMatrix(name)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(out)
	for _, row := range m.Rows {
		sortDestinations(row)
		n := len(row.Destinations)
		top := row.Destinations[0:k]
		bottom := row.Destinations[n-k-1 : n-1]
		for i, dest := range top {
			w.WriteString(row.Departure)
			w.WriteString("\t")
			w.WriteString(dest.Name)
			w.WriteString("\t")
			w.WriteString(bottom[k-i-1].Name)
			w.WriteString("\n")
		}
	}
	return w.Flush()
}

func main() {
	flag.IntVar(&k, "k", k, "")
	flag.Parse()
	name := flag.Arg(0)
	err := run(name)
	if err != nil {
		log.Fatal(err)
	}
}
