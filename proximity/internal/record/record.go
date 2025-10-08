package record

import (
	"encoding/csv"
	"errors"
	"io"
	"iter"
	"os"
	"strconv"
)

type Record struct {
	Index  int
	Name   string
	Vector []float64
}

func strings2record(strs []string) (*Record, error) {
	if len(strs) < 3 {
		return nil, errors.New("too few records in a line, require 3 or more")
	}
	index, err := strconv.Atoi(strs[0])
	if err != nil {
		return nil, err
	}
	name := strs[1]
	strs = strs[2:]
	vec := make([]float64, len(strs))
	for i, s := range strs {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		vec[i] = v
	}
	return &Record{
		Index:  index,
		Name:   name,
		Vector: vec,
	}, nil
}

func RecordIter(r io.Reader) iter.Seq2[*Record, error] {
	rr := csv.NewReader(r)
	rr.Comma = '\t'
	return func(yield func(*Record, error) bool) {
		for {
			strs, err := rr.Read()
			if err != nil {
				if errors.Is(err, io.EOF) {
					return
				}
				yield(nil, err)
				return
			}
			record, err := strings2record(strs)
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(record, nil) {
				return
			}
		}
	}
}

func LoadAll(name string) ([]Record, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var all []Record
	for r, err := range RecordIter(f) {
		if err != nil {
			return nil, err
		}
		all = append(all, *r)
	}
	return all, nil
}
