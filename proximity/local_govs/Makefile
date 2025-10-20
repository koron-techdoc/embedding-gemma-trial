all: accuracy_full.txt accuracy_partial.txt

truth_full.txt: local_govs_full.tsv
	go run ./gen_truth.go $< > $@

truth_partial.txt: local_govs_partial.tsv
	go run ./gen_truth.go $< > $@

accuracy_full.txt: truth_full.txt
	./cluster_local_govs.py -l $< > $@

accuracy_partial.txt: truth_partial.txt
	./cluster_local_govs.py -l $< > $@
