# Preprocessing Pipeline Removal

As demonstrated in our ablation tests, applying `ramanspy` preprocessing over raw data decreases the CNN's accuracy from 85.30% to 76.50%. Therefore, the `src/dataset.py` was modified to remove the pipeline (or isolate just the denoising step for the second experiment).

Here is the exact code change made to `src/dataset.py`:

```python
<<<< BEFORE
            pipeline = rp.preprocessing.Pipeline([
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
                rp.preprocessing.baseline.ASPLS(),
                rp.preprocessing.normalise.MinMax()
            ])
====
            pipeline = rp.preprocessing.Pipeline([
                rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)
            ])
>>>> AFTER
```

Later, even the Denoising step was proven to hurt accuracy (yielding 83.23% instead of 85.30%), so the entire pipeline block can be safely removed.
