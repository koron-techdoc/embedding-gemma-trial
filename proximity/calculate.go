package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/koron/embedding-gemma-demo/proximity/internal/record"
)

func calcDist(a, b record.Record) float64 {
	n := len(a.Vector)
	var sum float64
	for i := range n {
		d := a.Vector[i] - b.Vector[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func toRadian(degree float64) float64 {
	return degree / 180 * math.Pi
}

func geodicDistance(a, b record.Record) float64 {
	const (
		// World Geodetic System (GRS80)
		A = 6378137.0
		F = 1 / 298.257222101
		B = A * (1 - F)
	)

	lat1 := toRadian(a.Vector[0])
	lon1 := toRadian(a.Vector[1])
	lat2 := toRadian(b.Vector[0])
	lon2 := toRadian(b.Vector[1])

	phi1 := math.Atan2(B*math.Tan(lat1), A)
	phi2 := math.Atan2(B*math.Tan(lat2), A)

	x := math.Acos(math.Sin(phi1)*math.Sin(phi2) + math.Cos(phi1)*math.Cos(phi2)*math.Cos(lon2-lon1))

	drho := F / 8 * ((math.Sin(x)-x)*(math.Pow(math.Sin(phi1)+math.Sin(phi2), 2))/math.Pow(math.Cos(x/2), 2) - (math.Sin(x)+x)*(math.Pow(math.Sin(phi1)-math.Sin(phi2), 2))/math.Pow(math.Sin(x/2), 2))

	rho := A * (x + drho)

	return rho / 1000.0 // return in Kilometers
}

func run(name string) error {
	records, err := record.LoadAll(name)
	if err != nil {
		return err
	}
	n := len(records)

	scoreMatrix := make([][]float64, n)
	for i := range scoreMatrix {
		scoreMatrix[i] = make([]float64, n)
		scoreMatrix[i][i] = math.NaN()
	}

	calcFn := calcDist
	if len(records[0].Vector) == 2 {
		calcFn = geodicDistance
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			s := calcFn(records[i], records[j])
			scoreMatrix[i][j] = s
			scoreMatrix[j][i] = s
		}
	}

	// Header
	w := os.Stdout
	for _, r := range records {
		fmt.Fprint(w, "\t", r.Name)
	}
	// Body
	fmt.Fprint(w, "\n")
	for i, r := range records {
		fmt.Fprint(w, r.Name)
		for _, s := range scoreMatrix[i] {
			fmt.Fprintf(w, "\t%f", s)
		}
		fmt.Fprint(w, "\n")
	}

	return nil
}

func main() {
	flag.Parse()
	name := flag.Arg(0)
	err := run(name)
	if err != nil {
		log.Fatal(err)
	}
}
