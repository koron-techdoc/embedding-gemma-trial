package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/koron/embedding-gemma-demo/proximity/internal/record"
)

func run(name string) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	for r, err := range record.RecordIter(f) {
		if err != nil {
			return err
		}
		fmt.Printf("#%-2d  %s (%d)\n", r.Index, r.Name, len(r.Vector))
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
