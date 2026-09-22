# Future Work — Deep Learning Exploration

This folder contains exploratory LSTM/GRU experiments not included 
in the main capstone deliverables.

## What Was Tested
- LSTM and GRU recurrent networks on per-patient 48-hour sequences
- Two labeling strategies: detection (naive) vs prediction (6h lookahead)
- 3-fold GroupKFold cross-validation

## Key Finding
A substantial AUROC gap (0.934 to 0.567) emerged between the two 
labeling strategies. This gap quantifies the target-leakage effect 
of naive labeling. Under strict prediction framing, LSTM/GRU 
underperformed the temporal XGBoost model (AUROC 0.87).

## Why Not Included in Main Results
The temporal XGBoost model remains the primary contribution, 
outperforming all baselines and deep learning alternatives. 
Deep learning is documented here for completeness and as a 
recommendation for future work (Transformers, longer windows, 
multi-task learning).

## Files
- baseline_lstm_gru.py — initial LSTM/GRU exploration
- baseline_lstm_gru_option_c.py — detection vs prediction comparison
- option_c_log.txt — full training log with results
- dl_metrics*.json — metric summaries
- *_per_fold.csv — per-fold metrics
