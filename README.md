# Error correction back-end for speaker diarization
This repository provides a PyTorch implementation for DiaCorrect. Only training from scratch with CALLHOME/DIHARD3 are provided. More details will be updated soon.


# Usage
1. Prepare a pre-trained diarization system. See our EEND [implementation](https://github.com/BUTSpeechFIT/EEND).
2. Prepare initial speaker activity predictions. `bash infer_sap.sh`
3. Trainining. `python eend/train.py -c conf/train_conf.yaml`
4. Inference. `bash infer_step.sh`

# Result

| Dataset | DER |  MISS | FA | Conf. |
|:-------|:-----:|:--:|:----:|:--:|
|CH2 | 7.98 | 4.67 | 2.76 | 0.55 |
|DH3 eval | 12.47 | 7.11 | 4.64 | 0.72 |

# Contact
jyhan003@gmail.com






