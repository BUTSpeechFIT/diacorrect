## Error Correction Back-end for Speaker Diarization
This repository provides a PyTorch implementation for DiaCorrect by Brno University of Technology. By exploiting the interactions between the input recording and the initial system’s outputs, DiaCorrect can automatically refine the initial speaker activities to minimize the diarization errors. We focus on two scenarios:
- ground-truth labels in target-domain are not provided
- the pre-trained diarization system is taken as a black-box, only inference is available

For the first case, when using [EEND-EDA](https://arxiv.org/pdf/2005.09921.pdf) as an initial model, DiaCorrect trained on simulated data can achieve similar performance on CALLHOME 2-speaker subset as the initial model fine-tuned on real data. For the second case, when a small development set is available, DiaCorrect can be trained directly on it and also reach similar performance as the initial model fine-tuned.

More details can be found in [DiaCorrect: Error Correction Back-end For Speaker Diarization](https://arxiv.org/abs/2309.08377).


## Usage
Please replace the correct data path at first. Main steps are:
- prepare [simulated conversations](https://github.com/BUTSpeechFIT/EEND_dataprep).
- prepare a pre-trained EEND-EDA model. See our [implementation](https://github.com/BUTSpeechFIT/EEND). 
- perform data pruning for the first case. `bash scripts/data_pruning.sh`
- prepare initial speaker activity predictions. `bash scripts/infer_sap.sh`
- model trainining. `python diacorrect/train.py -c conf/train_*.yaml`
- inference. `bash scripts/infer_correct.sh`

## Results
We present DiaCorrect performance when training from scratch with real-labeled data.  
```
On CALLHOME Part2 (2 speakers)               On DiHARD3 CTS full eval
------------------------------------------------------------------------
System      DER   Miss   FA    Conf          DER   Miss   FA    Conf  
------------------------------------------------------------------------
EEND-EDA    8.62  3.40   4.53  0.69          19.58  3.94  14.62  1.02
    + FT    7.88  5.02   2.18  0.68          12.76  8.03  3.88   0.85
------------------------------------------------------------------------
DiaCorrect  7.98  4.67   2.76  0.55          12.47  7.11  4.64   0.72
------------------------------------------------------------------------

```
## Citation
If you found this work helpful, please consider citing
```
@article{han2023diacorrect,
      title={DiaCorrect: Error Correction Back-end For Speaker Diarization}, 
      author={Jiangyu Han and Federico Landini and Johan Rohdin and Mireia Diez and Lukas Burget and Yuhang Cao and Heng Lu and Jan Cernocky},
      journal={arXiv preprint arXiv:2309.08377},
      year={2023},
}
```

## Contact
If you have any comments or questions, please contact jyhan003@gmail.com or landini@fit.vutbr.cz.






