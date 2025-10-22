package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
)

type LGov struct {
	Code       string
	Prefecture string
	Name       string
}

func run(name string) error {
	f, err := os.Open(name)
	if err != nil {
		log.Printf("open %s failed: %s", name, err)
		return err
	}
	r := csv.NewReader(f)
	r.Comma = '\t'

	// load entries.
	lgovs := map[string]LGov{}
	name2codes := map[string][]string{}

	for {
		records, err := r.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		code := records[0]
		pref := records[1]
		name := records[2]
		entry := LGov{
			Code:       code,
			Prefecture: pref,
			Name:       name,
		}
		lgovs[code] = entry
		name2codes[name] = append(name2codes[name], code)
	}

	// Show duplication
	for name, codes := range name2codes {
		fmt.Printf("%s", name)
		for _, code := range codes {
			fmt.Printf("\t%s", lgovs[code].Prefecture)
		}
		fmt.Println()
	}

	return nil
}

func main() {
	flag.Parse()
	name := "local_govs_full.tsv"
	if flag.NArg() > 0 {
		name = flag.Arg(0)
	}
	err := run(name)
	if err != nil {
		log.Fatal(err)
	}
}
