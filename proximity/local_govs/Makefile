all: train_full.tsv train_partial.tsv

truth_full.txt: local_govs_full.tsv
	go run ./gen_truth.go $< > $@

truth_partial.txt: local_govs_partial.tsv
	go run ./gen_truth.go $< > $@

accuracy_full.txt: truth_full.txt
	./cluster_local_govs.py -l $< > $@

accuracy_partial.txt: truth_partial.txt
	./cluster_local_govs.py -l $< > $@

train_full.tsv: accuracy_full.txt
	go run ./gen_train.go $< > $@

train_partial.tsv: accuracy_partial.txt
	go run ./gen_train.go $< > $@
