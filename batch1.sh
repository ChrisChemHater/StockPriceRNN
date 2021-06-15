#!/bin/bash
pkurun-g4c 1 1 python hyper_params.py --dataset aapl --model lstm --mode model --device "cuda:0"
pkurun-g4c 1 1 python hyper_params.py --dataset aapl --model gru  --mode model --device "cuda:0"
pkurun-g4c 1 1 python hyper_params.py --dataset msft --model lstm --mode model --device "cuda:0"
pkurun-g4c 1 1 python hyper_params.py --dataset msft --model gru  --mode model --device "cuda:0"
