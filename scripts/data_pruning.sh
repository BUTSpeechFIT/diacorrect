#!/bin/bash

set -eu

# The simulated training data contains several thousands of files. 
# Running scoring on all of them at once is impossible. This script runs scoring 
# in batches of few files at a time and the variable 'subset_end' denotes that amount.
# Once all files are scored, those with DER between 'low_der' and 'high_der' are selected.


# input rttm 
rttm_eval_dir=              # data directory of evaluated rttm file 
rttm_oracle=                # oracle rttm file 
# output dir
out_dir=                

#---------------data split and evaluation---------------------#
subsets_dir=$out_dir/subsets
rttm_eval_subset=$subsets_dir/rttm_eval_subset
rttm_oracle_subset=$subsets_dir/rttm_oracle_subset

# data split
# subset number/index could be changed based on the hardware and dataset
subset_start=1
subset_end=32
subset_gap=$subset_end
subset_num=1

#------------------------data pruning--------------------------#
all_score_dir=$out_dir/all_score
pruned_dir=$out_dir/pruned
data_src_dir=               # Kaldi data directory for original dataset
# change threshold of DER if you want
low_der=12.0                
high_der=40.0
pruned_info=low${low_der}_high${high_der}_pruned

# dscore
dscore_dir=dscore
score_end=`expr $subset_end + 2`

mkdir -p $out_dir
mkdir -p $subsets_dir
mkdir -p $all_score_dir
mkdir -p $pruned_dir

stage=0
if [ $stage -le 0 ]; then
    echo "stage 0: prepare data.lst and data.path"
    find $rttm_eval_dir -iname "*.rttm" > $subsets_dir/${data}_eval.path
    cat $subsets_dir/${data}_eval.path | awk -F '/' '{print $NF}' | cut -d "." -f 1 > $subsets_dir/${data}_eval.lst
fi 

if [ $stage -le 1 ]; then
    echo "stage 1: data split and evaluation"
    eval_rttm_lst=$subsets_dir/${data}_eval.lst
    eval_rttm_path=$subsets_dir/${data}_eval.path
    all_num=`cat $eval_rttm_lst | wc -l`
    while (($subset_start <= $all_num))
    do 
        echo "****************************"
        echo "subset_start: $subset_start"
        echo "subset_end: $subset_end"
        echo "subset_num: $subset_num"

        subset_out=${rttm_eval_subset}${subset_num}

        mkdir -p ${subset_out} 
        sed -n "${subset_start}, ${subset_end}p" $eval_rttm_lst > ${subset_out}/data.lst
        cat ${subset_out}/data.lst | awk '{printf("grep %s %s\n", $0, "'$eval_rttm_path'")}' | sh > ${subset_out}_eval_rttm_path
        cat ${subset_out}_eval_rttm_path | awk '{printf("cp %s %s\n", $0, "'$subset_out'")}' | sh
        cat ${subset_out}/data.lst | awk '{printf("grep %s %s\n", $0, "'$rttm_oracle'")}' | sh > ${rttm_oracle_subset}${subset_num}

        python ${dscore_dir}/score.py -r ${rttm_oracle_subset}${subset_num} -s ${subset_out}/*.rttm --collar 0.25 > ${subset_out}/result_collar0.25
        sed -n "3, ${score_end}p" ${subset_out}/result_collar0.25 > ${subset_out}/result_collar0.25_score
        subset_start=`expr $subset_start + $subset_gap`
        subset_end=`expr $subset_end + $subset_gap`
        subset_num=`expr $subset_num + 1`
    done
    echo "combine subset scoring..."
    cat ${rttm_eval_subset}*/*_score > ${all_score_dir}/${data}_all_score
    cat ${all_score_dir}/${data}_all_score | awk '{printf("%s %s\n", $1, $2)}' > ${all_score_dir}/${data}_all_der
fi

if [ $stage -le 2 ]; then
    echo "stage 2: data selection"
    python scripts/data_selection.py ${all_score_dir}/${data}_all_der $pruned_dir --low-der $low_der --high-der $high_der
    for data in wav.scp sap.scp reco2dur rttm segments utt2spk spk2utt 
    do
        echo "processing ${data}..."
        cat ${pruned_dir}/${pruned_info} | awk '{printf("grep %s %s\n", $1, "'${data_src_dir}/${data}'")}' | sh > ${pruned_dir}/${data}
    done
fi 