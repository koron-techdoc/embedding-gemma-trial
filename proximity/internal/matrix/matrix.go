package matrix

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type Matrix struct {
	Rows []Row
}

type Row struct {
	Departure    string
	Destinations []Destination
}

type Destination struct {
	Name     string
	Distance float64
}

func toDestinations(records, labels []string) ([]Destination, error) {
	out := make([]Destination, len(records))
	for i, s := range records {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		out[i] = Destination{
			Name:     labels[i],
			Distance: v,
		}
	}
	return out, nil
}

func LoadMatrix(name string) (*Matrix, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '\t'
	labels, err := r.Read()
	if err != nil {
		return nil, err
	}
	labels = labels[1:]

	m := &Matrix{}
	for {
		records, err := r.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		dests, err := toDestinations(records[1:], labels)
		if err != nil {
			return nil, err
		}
		m.Rows = append(m.Rows, Row{
			Departure:    records[0],
			Destinations: dests,
		})
	}
	return m, nil
}

func Write(out io.Writer, m *Matrix) error {
	w := bufio.NewWriter(out)
	for _, row := range m.Rows {
		io.WriteString(w, row.Departure)
		for _, dest := range row.Destinations {
			fmt.Fprintf(w, "\t%s %f", dest.Name, dest.Distance)
		}
		io.WriteString(w, "\n")
	}
	return w.Flush()
}

func toDestinations2(records []string) ([]Destination, error) {
	out := make([]Destination, len(records))
	for i, s := range records {
		x := strings.Index(s, " ")
		if x < 0 {
			return nil, errors.New("format error: whitespace not found")
		}
		name := s[:x]
		v, err := strconv.ParseFloat(s[x+1:], 64)
		if err != nil {
			return nil, err
		}
		out[i] = Destination{
			Name:     name,
			Distance: v,
		}
	}
	return out, nil
}

func LoadFile(name string) (*Matrix, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '\t'

	m := &Matrix{}
	for {
		records, err := r.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		dests, err := toDestinations2(records[1:])
		if err != nil {
			return nil, err
		}
		m.Rows = append(m.Rows, Row{
			Departure:    records[0],
			Destinations: dests,
		})
	}
	return m, nil
}
