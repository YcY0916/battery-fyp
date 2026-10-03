# Shard files

One run per line: `<config> <seed>[,<seed>...]`; `stage:` sets the stage; `#` starts a comment.
Used with `python src/run_shard.py --shard-file shards/<file> --platform <platform> --skip-existing`.

| File | Platform | Runs |
|---|---|---|
| `step2_colab.txt` | Colab T4 | L1 x W60/120/180 x H32/64/128 (9) |
| `step2_kaggle_gpu0.txt` | Kaggle T4 #0 | L1_W360 x H32/64/128, L2_W240 x H32/64/128 (6) |
| `step2_kaggle_gpu1.txt` | Kaggle T4 #1 | L1_W240 and L1_W300 x H32/64/128, L2_W180 x H32/64/128 (9) |
| `step2_laptop.txt` | Laptop RTX 4050 | L2 x W300/360 x H32/64/128 (6) |
| `probe.txt` | every platform | timing probe: L2_W360_H128 and L1_W60_H32, 3 epochs, experiment `timing_v2` |

`step3_candidates.txt` (the 6 configs chosen by the author) and the `step3_*.txt` shard files are
added after step 2. The split of runs between platforms does not change any training setting; every
run records the platform it ran on.
