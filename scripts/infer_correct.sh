#!/bin/bash

set -eu

##################################
med=11
collar=0.25
data_type=callhome_part2_spk2
infer_data_dir=             # Kaldi data directory for the evaluation set
infer_sap_scp=sap/callhome_part2_2spk/sap.scp
ref_rttm_dir=               # data directory of oracle rttm file 

##################################
# med=1
# collar=0
# data_type=DH20ctsfulleval
# infer_data_dir=            # Kaldi data directory for the evaluation set
# infer_sap_scp=sap/DIHARD2020eval.full.cts/sap.scp
# ref_rttm_dir=              # data directory of oracle rttm file    

#score
dscore_dir=dscore
exp_dir=                     # exp dir
model_path=${exp_dir}/models

start_ep=0
avg_num=10
end_ep=50 

bias=0

while (($start_ep < $end_ep))
do
    epochs=$start_ep-`expr $start_ep + $avg_num`
    echo "epochs ${epochs}..."
    rttms_dir=${exp_dir}/infer_calibration_${bias}/${data_type}
    python diacorrect/infer.py -c conf/infer_common.yaml \
                --infer-data-dir ${infer_data_dir} --infer-sap-scp ${infer_sap_scp} \
                --bias ${bias} --epochs ${epochs} --models-path ${model_path} \
                --rttms-dir ${rttms_dir} --median-window-length ${med}
    start_ep=`expr $start_ep + $avg_num`
    # score
    out_dir=$(find ${PWD}/${rttms_dir} -iname "*.rttm" | head -n 1 | awk -F '/epochs' '{print $1}')
    out_dir=${out_dir}/epochs${epochs}/detection_thr0.5/median${med}
    python ${dscore_dir}/score.py -r ${ref_rttm_dir}/rttm -s ${out_dir}/rttms/*.rttm --collar ${collar} > ${out_dir}/result_collar${collar}
done
