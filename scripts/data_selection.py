#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:46:25 2023

@author: jyhan
"""

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input score file")
    parser.add_argument(
        "output", type=str, help="Path to the output dir"
    )
    parser.add_argument(
        "--low-der", type=float, default=12.0, help="lower DER% threshold"
    )
    parser.add_argument(
        "--high-der", type=float, default=40.0, help="higher DER% threshold"
    )
    args = parser.parse_args()

    der_list = []
    data_filtered = []
    lines = open(args.input, 'r').readlines()
    for line in lines:
        der_list.append(float(line.split()[-1]))
        if der_list[-1] >= args.low_der and der_list[-1] <= args.high_der:
            data_filtered.append(der_list[-1])

    outfile = f'low{args.low_der}_high{args.high_der}_pruned'
    with open(os.path.join(args.output, outfile), 'w') as f:
        for line in lines:
            name, der = line.split()
            if float(der) >= args.low_der and float(der) <= args.high_der:
                f.write(name + ' ' + der + '\n')




        