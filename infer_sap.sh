#!/bin/bash

set -eu

infer_conf=conf/infer_sap_common.yaml
eval_dir=/mnt/matylda4/landini/diarization/EEND/evaluation_sets         # CH1, CH2, DH3 dev, DH3 eval
out_dir=sap    # initial sap output

# Prepare initial speaker activity
for data in callhome_part1_2spk callhome_part2_2spk DIHARD2020dev.full.cts DIHARD2020eval.full.cts;
do
	echo "Prepare $data"
	sap_out=$out_dir/$data
	mkdir -p $sap_out
	python eend/infer_sap.py -c $infer_conf --infer-data-dir $eval_dir/$data --out-dir $sap_out	
	# prepare scp file
	find $PWD/$sap_out | grep '\.h5' | awk -F '/' '{print $NF}' | cut -d "." -f 1 > $sap_out/sap.list
	find $PWD/$sap_out | grep '\.h5' > $sap_out/sap.path
	paste $sap_out/sap.list $sap_out/sap.path > $sap_out/sap.scp
done
