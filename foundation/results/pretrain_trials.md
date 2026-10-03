| run | preset | params | epochs | hours | best epoch | probe | k-NN | logreg | raw k-NN | G1 | RankMe | source-ID | airborne (kNN/LR) | rruff (kNN/LR) | end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261003-034431_tiny_trial | tiny | 0.92M | 50 | 0.6 | 40 | 0.636 | 0.598 | 0.675 | 0.625 | FAIL | 57.0 | 0.931 | 0.84/0.92 | 0.36/0.43 | finished 50/50: best probe score at epoch 40 |
| 20261003-042345_small_trial | small | 4.99M | 50 | 1.1 | 40 | 0.672 | 0.644 | 0.700 | 0.625 | PASS | 99.4 | 0.941 | 0.87/0.91 | 0.42/0.49 | finished 50/50: best probe score at epoch 40 |
| 20261003-052918_base_trial | base | 14.57M | 50 | 1.9 | 30 | 0.672 | 0.635 | 0.710 | 0.625 | PASS | 125.3 | 0.950 | 0.87/0.93 | 0.40/0.49 | finished 50/50: best probe score at epoch 30 |

Recommended size: small (20261003-042345_small_trial). small, base are statistically indistinguishable on the probe (within 0.03 of the best score 0.672), so the smaller one is recommended.
