#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini, Jiangyu Han)
# Licensed under the MIT license.

from common_utils.ckpt_utils import (
    average_checkpoints,
    get_model,
)

from common_utils.dataset_eend import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from os.path import join
from torch.utils.data import DataLoader
from types import SimpleNamespace
import logging
import numpy as np
import os
import random
import torch
import yamlargparse

import h5py
from typing import Any, Dict, List, Tuple

EPS = 1e-8
def sigmoid_inverse(x):
    return torch.log((x + EPS) / (1 - x + EPS))

def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]:
    return {'xs': [x for x, _, _ in batch],
            'ts': [t for _, t, _ in batch],
            'names': [r for _, _, r in batch]}

def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
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

    Y, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND inference')
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
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--out-dir', type=str,
                        help='output directory for SAP.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--osd-loss-weight', default=0.0, type=float)

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument('--attractor-loss-ratio', default=1.0,
                                type=float, help='weighting parameter')
    attractor_args.add_argument('--attractor-encoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--attractor-decoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--estimate-spk-qty', default=-1, type=int)
    attractor_args.add_argument('--estimate-spk-qty-thr',
                                default=-1, type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', default=False, type=bool,
        help='If True, avoid backpropagation on attractor loss')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
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

    out_dir = args.out_dir

    for i, batch in enumerate(infer_loader):
        input = torch.stack(batch['xs']).to(args.device)
        name = batch['names'][0]
        with torch.no_grad():
            y_pred = model.estimate_sequential(input, args)[0]
            print(f'name: {name} | {y_pred.shape}')
        # remove sigmoid
        y_pred = sigmoid_inverse(y_pred)
        outdata = y_pred.cpu().detach().numpy()
        out_filename = join(out_dir, f"{name}.h5")
        with h5py.File(out_filename, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
