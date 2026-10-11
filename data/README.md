# Benchmark data — not redistributed

The three real-data benchmarks use public datasets released by other authors, which we do not redistribute.
**The directories below are shipped empty.** Download the files from the sources given here and put them in
the directory shown, keeping the file names exactly as listed. Nothing else in the package needs to change:
the scripts resolve these paths relative to their own location.

The synthetic benchmarks (demand, low-dimensional ATE/ATT/CATE, the bridge-route scenarios) generate their
data in-process and need nothing here.

A script run before the files are in place fails with `FileNotFoundError` naming the file it wanted; that
is the expected behaviour of an empty directory, not a broken package.

## Abortion & Crime, and grade retention (Deaner IM / IR)

Source: **https://github.com/yuchen-zhu/kernel_proxies**

```
data/abortion/train/main_ab_seed{0..9}.npz                 10 files: A, Y, Z, W per benchmark seed
data/abortion/test/do_A_ab_seed{0..9}.npz                  10 files: the do(A) grid and its ground truth
data/deaner/IM_80_N/main_edu_IM_80_seed{100,200,...,1000}.npz     10 files
data/deaner/IM_80_N/do_A_edu_IM_80_seed{100,200,...,1000}.npz     10 files
data/deaner/IR_80_N/main_edu_IR_80_seed{100,200,...,1000}.npz     10 files
data/deaner/IR_80_N/do_A_edu_IR_80_seed{100,200,...,1000}.npz     10 files
```

Read by `read_legalized_abortion_and_crime_dataset` and `read_deaner_dataset` in
`src/simulation_utils/data_generating_processes.py`. Used by the Abortion and Deaner simulation scripts and
their kernel counterparts in `Simulations/DRKPV/`.

## dSprite

Source: **https://github.com/liyuan9988/DeepFeatureProxyVariable**

```
data/dsprite/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz    the public dSprites release (~19 MB)
data/dsprite/dsprite_mat.npy                                   the 4096 x 10 projection matrix B
```

The `.npz` is the standard dSprites dataset; `dsprite_mat.npy` is the fixed random projection that the
proceedings variant of the benchmark uses. Read by the `dSprite_ProxyVariable_Dataset*` classes in
`src/simulation_utils/data_generating_processes.py`, used by the dSprite V2 / V3 simulation scripts, the
dSprite kernel baseline, and the proxy-informativeness ablation.
