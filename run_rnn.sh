#!/bin/bash
# Script to train and evaluate the RamanLSTM (RNN) model

echo "Training RamanLSTM on Reference Data..."
.venv/bin/python main.py --model-type rnn --epochs 15 --finetune-epochs 5 > results/pure_rnn.log 2>&1

echo "Done! Logs saved to results/pure_rnn.log"
