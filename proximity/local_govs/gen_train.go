package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"slices"
	"strconv"
	"strings"
)

type Estimation struct {
	Name  string
	K     int
	Hit   int
	TopK  []string
	NextK []string
	Truth []string
}

type Accuracy struct {
	Estimations []*Estimation
	Accuracy    float64
}

func LoadAccuracy(name string) (*Accuracy, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '\t'
	acc := Accuracy{}

	for {
		r.FieldsPerRecord = 0
		records, err := r.Read()
		if err != nil {
			return nil, err
		}
		// Parse accuracy value
		if len(records) == 1 {
			const accuracyLeader = "accuracy: "
			x := strings.Index(records[0], accuracyLeader)
			if x < 0 {
				return nil, errors.New("not found accuracy leader")
			}
			acc.Accuracy, err = strconv.ParseFloat(records[0][len(accuracyLeader):], 64)
			if err != nil {
				return nil, err
			}
			break
		}
		khit := strings.SplitN(records[1], "/", 3)
		k, err := strconv.Atoi(khit[1])
		if err != nil {
			return nil, err
		}
		hit, err := strconv.Atoi(khit[0])
		if err != nil {
			return nil, err
		}
		acc.Estimations = append(acc.Estimations, &Estimation{
			Name:  records[0],
			K:     k,
			Hit:   hit,
			TopK:  strings.Split(records[2], ","),
			NextK: strings.Split(records[3], ","),
			Truth: strings.Split(records[4], ","),
		})
	}

	return &acc, nil
}

func calcNegatives(top, next, truth []string) []string {
	negatives := make([]string, 0, len(truth))
	for _, curr := range append(top, next...) {
		if slices.Contains(truth, curr) {
			continue
		}
		negatives = append(negatives, curr)
		if len(negatives) == len(truth) {
			break
		}
	}
	return negatives
}

func Run(name string) error {
	acc, err := LoadAccuracy(name)
	if err != nil {
		return err
	}
	for _, est := range acc.Estimations {
		negatives := calcNegatives(est.TopK, est.NextK, est.Truth)
		for k := 0; k < est.K; k++ {
			pos := est.Truth[k]
			neg := negatives[k]
			fmt.Printf("%s\t%s\t%s\n", est.Name, pos, neg)
		}
	}
	return nil
}

func main() {
	flag.Parse()
	name := "accuracy_full.txt"
	if flag.NArg() > 0 {
		name = flag.Arg(0)
	}
	err := Run(name)
	if err != nil {
		log.Fatal(err)
	}
}
