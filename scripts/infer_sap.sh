#!/bin/bash

set -eu

infer_conf=conf/infer_sap.yaml
eval_dir=        # Kaldi data directory for the evaluation set
out_dir=sap      # initial sap output

# Prepare initial speaker activity
# if you want get SAP for simulated data, just put it into the list
for data in callhome_part1_2spk callhome_part2_2spk DIHARD2020dev.full.cts DIHARD2020eval.full.cts;
do
	echo "Prepare $data"
	sap_out=$out_dir/$data
	mkdir -p $sap_out
	python diacorrect/infer_sap.py -c $infer_conf --infer-data-dir $eval_dir/$data --out-dir $sap_out	
	# prepare scp file
	find $PWD/$sap_out | grep '\.h5' | awk -F '/' '{print $NF}' | cut -d "." -f 1 > $sap_out/sap.list
	find $PWD/$sap_out | grep '\.h5' > $sap_out/sap.path
	paste $sap_out/sap.list $sap_out/sap.path > $sap_out/sap.scp
done
