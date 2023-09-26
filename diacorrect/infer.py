#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini, Jiangyu Han)
# Licensed under the MIT license.

from common_utils.ckpt_utils import (
    average_checkpoints,
    get_model,
)
from common_utils.dataset_correct import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from os.path import join
from pathlib import Path
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import TextIO
import logging
import numpy as np
import os
import random
import torch
import yamlargparse

import torch.nn.functional as F

def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        args.infer_sap_scp,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=0,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y, _, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader

def hard_labels_to_rttm(
    labels: np.ndarray,
    id_file: str,
    rttm_file: TextIO,
    frameshift: float = 10
) -> None:
    """
    Transform NfxNs matrix to an rttm file
    Nf is the number of frames
    Ns is the number of speakers
    The frameshift (in ms) determines how to interpret the frames in the array
    """
    if len(labels.shape) > 1:
        # Remove speakers that do not speak
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    if len(labels.shape) > 1:
        n_spks = labels.shape[1]
    else:
        n_spks = 1
    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]
        # Add final mark if needed
        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([
                end_indices,
                labels.shape[0] - 1])
        assert len(ini_indices) == len(end_indices), \
            "Quantities of start and end of segments mismatch. \
            Are speaker labels correct?"
        n_segments = len(ini_indices)
        for index in range(n_segments):
            spk_list.append(spk)
            ini_list.append(ini_indices[index])
            end_list.append(end_indices[index])
    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 " +
            f"{round(ini * frameshift / 1000, 3)} " +
            f"{round((end - ini) * frameshift / 1000, 3)} " +
            f"<NA> <NA> spk{spk} <NA> <NA>\n")

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def postprocess_output(
    probabilities,
    subsampling: int,
    threshold: float,
    median_window_length: int
) -> np.ndarray:
    thresholded = probabilities > threshold
    thresholded = 1.0 * thresholded
    filtered = np.zeros(thresholded.shape)
    for spk in range(filtered.shape[1]):
        filtered[:, spk] = medfilt(
            thresholded[:, spk],
            kernel_size=median_window_length)
        # filtered[:, spk] = thresholded[:, spk]

    probs_extended = np.repeat(filtered, subsampling, axis=0)
    return probs_extended

def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='DiaCorrect inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--infer-sap-scp', help='inference sap scp.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='DiaCorrect',
                        help='Type of model (for now only DiaCorrect)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--bias', default=0.0, type=float, help='calibration bias')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--osd-loss-weight', default=0.0, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    logging.info(args)

    infer_loader = get_infer_dataloader(args)

    if args.gpu >= 1:
        gpuid = use_single_gpu(args.gpu)
        logging.info('GPU device {} is used'.format(gpuid))
        args.device = torch.device("cuda")
    else:
        gpuid = -1
        args.device = torch.device("cpu")

    model = get_model(args)
    model = average_checkpoints(
        args.device, model, args.models_path, args.epochs)
    model.eval()

    out_dir = join(
        args.rttms_dir,
        f"epochs{args.epochs}",
        f"detection_thr{args.threshold}",
        f"median{args.median_window_length}",
        "rttms"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(infer_loader):
        input = torch.stack(batch['xs']).to(args.device)
        sap = torch.stack(batch['sap']).to(args.device)
        # calibration
        sap = sap + args.bias

        name = batch['names'][0]
        with torch.no_grad():
            y_pred = model(input, sap, sigmoid=True)[0]
            print(f'i={i+1} | name: {name} | {y_pred.shape}')
        y_pred = y_pred.cpu().detach().numpy()
        post_y = postprocess_output(
            y_pred, args.subsampling,
            args.threshold, args.median_window_length)
        rttm_filename = join(out_dir, f"{name}.rttm")
        with open(rttm_filename, 'w') as rttm_file:
            hard_labels_to_rttm(post_y, name, rttm_file)
