#!/bin/bash

set -eu

##################################
med=11
collar=0.25
data_type=callhome_part2_spk2
infer_data_dir=/YOUR_EVAL_DIR/callhome_part2_2spk
infer_sap_dir=/YOUR_SAP_DIR/callhome_part2_2spk/sap.scp
ref_rttm_dir=/YOUR_EVAL_DIR/callhome_part2_2spk

##################################
# med=1
# collar=0
# data_type=DH20ctsfulleval
# infer_data_dir=/YOUR_EVAL_DIR/DIHARD2020eval.full.cts
# infer_sap_dir=/SAP_YOUR_SAP_DIRDIR/DH20ctsfulleval/sap.scp
# ref_rttm_dir=/YOUR_EVAL_DIR/DIHARD2020eval.full.cts

#score
dscore_dir=/YOUR_SCORE_DIR/dscore
exp_dir=exp/train_scratch_callhome_lr_1e-5
# exp_dir=exp/train_scratch_dihard_lr_1e-4
model_path=${exp_dir}/models

start_ep=0
avg_num=4
end_ep=4

shift_val=0

while (($start_ep < $end_ep))
do
    epochs=$start_ep-`expr $start_ep + $avg_num`
    echo "epochs ${epochs}..."
    rttms_dir=${exp_dir}/infer_shift_${shift_val}_step/${data_type}
    python eend/infer_correct_shift.py -c conf/infer_correct_common.yaml \
                --infer-data-dir ${infer_data_dir} --infer-sap-dir ${infer_sap_dir} \
                --shift ${shift_val} \
                --epochs ${epochs} --models-path ${model_path} \
                --rttms-dir ${rttms_dir} --median-window-length ${med}
    start_ep=`expr $start_ep + $avg_num`
    # score
    out_dir=$(find ${PWD}/${rttms_dir} -iname "*.rttm" | head -n 1 | awk -F '/epochs' '{print $1}')
    out_dir=${out_dir}/epochs${epochs}/timeshuffleTrue/spk_qty2_spk_qty_thr-1.0/detection_thr0.5/median${med}
    python ${dscore_dir}/score.py -r ${ref_rttm_dir}/rttm -s ${out_dir}/rttms/*.rttm --collar ${collar} > ${out_dir}/result_collar${collar}
done
