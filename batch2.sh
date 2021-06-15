#!/bin/bash
pkurun-g4c 1 1 python hyper_params.py --dataset aapl --model lstm --mode train --device "cuda:0"
pkurun-g4c 1 1 python hyper_params.py --dataset aapl --model gru  --mode train --device "cuda:0"
pkurun-g4c 1 1 python hyper_params.py --dataset msft --model lstm --mode train --device "cuda:0"
pkurun-g4c 1 1 python hyper_params.py --dataset msft --model gru  --mode train --device "cuda:0"
