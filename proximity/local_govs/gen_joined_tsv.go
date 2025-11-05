package main

// `{キー}\t{値}` 形式のTSVストリーム、キーの重複がありうるもの、を読みこんで
// `{キー}\t{値1}\t{値2}\t...\t{値n} の形式のTSVストリーム=に変換する

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

var (
	keyList  []string
	key2vals = map[string][]string{}
)

func add(key string, values ...string) {
	oldValues, ok := key2vals[key]
	if !ok {
		keyList = append(keyList, key)
	}
	key2vals[key] = append(oldValues, values...)
}

func run(w io.Writer, r io.Reader) error {
	cr := csv.NewReader(r)
	cr.Comma = '\t'
	for {
		cr.FieldsPerRecord = 0
		records, err := cr.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		add(records[0], records[1])
	}

	for _, key := range keyList {
		values := key2vals[key]
		fmt.Fprintf(w, "%s\t%s\n", key, strings.Join(values, "\t"))
	}

	return nil
}

func main() {
	err := run(os.Stdout, os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
}
