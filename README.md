# Adaptive Doubly Robust Proxy Causal Learning

Code for the neural mean-embedding proxy causal learning estimators — the outcome bridge
(`OutcomeNet`), the density-ratio-free treatment bridge (`TreatmentNet`), and the doubly robust
estimator (`DR-PCL-Net`) — together with the simulation scripts behind every experiment in the paper
and the analysis notebooks that produce its tables and figures.

Bariscan Bozkurt, Alexandre Galashov, Dimitri Meunier, Zikai Shen, Arthur Gretton, Houssam Zenati.

> Paper: *Adaptive Doubly Robust Proxy Causal Learning* — arXiv preprint (link to be added).

## Layout

```
src/                            the method (neural_causal_learning/), our kernel implementations
                                (causal_learning/), data-generating processes and network builders
                                (simulation_utils/), shared utilities (torch_utils/, python_utils/)
Simulations/
  DR-NME-PCL-Dens-Ratio-Free/   one script per benchmark: Abortion, Deaner, Demand, dSprite (V2),
                                dSprite V3, synthetic ATT, synthetic CATE
  BridgeRoutePreference/        the bridge-route stress tests (Scenario B, Scenario O)
  DRKPV/, KPV/                  the kernel baselines (KPV, KAP, DRKPV; exact and Nyström)
  AnalyzeSimulationResults/     the analysis notebooks, executed, with the figures in Figures/
  Results/                      where the scripts write their CSVs (none are shipped)
data/                           the benchmark directory layout, shipped empty, and a README with the
                                download links (the data itself is not redistributed here)
```

No result files, logs, or cluster submission scripts are included: the notebooks ship with the outputs
they produced, and every run writes fresh CSVs into `Simulations/Results/`.

## Requirements

Python 3.12 and the packages in `requirements.txt` (`pip install -r requirements.txt`). A GPU is used when
available but is never required.

The three real-data benchmarks (Abortion & Crime, Deaner IM/IR, dSprite) need public datasets released by
other authors, which we do not redistribute: the directories under `data/` are shipped empty, and
`data/README.md` gives the download links and the exact file names each one expects. The synthetic
benchmarks need nothing. Both sources are the ones the original papers use:

- Abortion & Crime, and Deaner IM/IR: https://github.com/yuchen-zhu/kernel_proxies
- dSprite: https://github.com/liyuan9988/DeepFeatureProxyVariable

## Running a simulation

Every script is self-contained, exposes its hyperparameters through `argparse`, and writes one CSV per run
with its configuration recorded as columns. Run a script from its own directory:

```bash
cd Simulations/DR-NME-PCL-Dens-Ratio-Free
python Demand_CrossFitting_Simulation.py --n_sample 4000 --data_seeds 0 --n_repeats 3
python Demand_CrossFitting_Simulation.py --help     # all options and their defaults
```

Data seeds and repeats are separate: `--data_seeds` selects the data sets and `--n_repeats` the independent
re-runs on each (the train seed is a deterministic function of both). The reported grids use 10 data seeds
and 3 repeats, which is a large amount of compute; a single `--data_seeds 0 --n_repeats 1` run reproduces
one cell and is the quickest way to check an installation.

Every run records its treatment-bridge normalisation arm in a `normalization` column, but how many arms one
invocation produces differs by script. Demand, Deaner and synthetic ATT default to `--normalization all` and
write `none`, `posthoc` and `kkt` from a single run off one shared outcome bridge, so their ablation tables
come from the same runs as their main tables. Synthetic CATE has no `posthoc` arm (its `--help` explains why)
and defaults to `both` = `none` + `kkt`. Abortion runs one arm per invocation, so its ablation needs two
runs (below). The dSprite scripts take `--arms`, and the reported grids run `kkt-mean` only.

### Reproducing the reported numbers

Each row gives the flags the reported grid used (beyond `--data_seeds` / `--n_repeats 3`) and the notebook
that reads the result. The `--tag` value matters: most notebooks filter on it, so a different tag makes the
rows disappear rather than disagree. Two loads do **not** filter by tag — the synthetic ATT table (notebook 7)
and the dSprite V3 kernel rows (notebook 11) — so keep exploratory runs out of those two result folders or
they will be pooled into the reported numbers.

| experiment | command (from its script's directory) | notebook |
|---|---|---|
| Abortion & Crime | `Abortion_CrossFitting_Simulation.py --normalization kkt --third_epochs 10 --tag kkt`, and for the ablation's second arm `--normalization posthoc --third_epochs 10 --tag baseline` | 1 |
| Deaner IM | `Deaner_CrossFitting_Simulation.py --third_center 1 --third_head rff_mlp --tag center_rff` | 4 |
| Deaner IR | `Deaner_CrossFitting_Simulation.py --deaner_id IR --third_center 1 --third_head rff_mlp --tag center_rff_ir` | 4 |
| Demand (size sweep) | `Demand_CrossFitting_Simulation.py --n_sample {2000,5000,10000,15000,20000} --tm_third_epochs 200 --dr_third_epochs 200 --tag sizesweep` | 2 |
| dSprite V2 (size sweep) | `dSprite_CrossFitting_Simulation.py --n_sample {2000,...,20000} --om_trainer v2 --om_arch sn --y_scale raw --z_scale raw --om_batch 6000 --om_epochs 20 --om_stage1_iter 20 --om_lr 1e-3 --om_weight_decay 0.1 --om_v2_bn_train_pred 1 --om_reg1_lo 1 --om_reg21_lo 1 --om_reg2_lo 10 --om_reg2_hi 10 --third_loss huber --om_both 0 --arms kkt-mean --dr_third_lr 1e-4 --dr_third_epochs 200 --tag v2lam1dr` | 9 |
| dSprite V3 (size sweep) | `dSprite_V3_CrossFitting_Simulation.py --n_sample {2000,...,20000} --om_trainer v2 --om_arch sn --y_scale raw --z_scale raw --om_batch 6000 --om_epochs 20 --om_stage1_iter 20 --om_lr 1e-3 --om_weight_decay 0.1 --om_v2_bn_train_pred 1 --om_reg1_lo 1 --om_reg21_lo 1 --om_reg2_lo 10 --om_reg2_hi 10 --third_loss huber --om_both 0 --arms kkt-mean --tag v3lam1` | 11 |
| Synthetic ATT (size sweep) | `Synthetic_ATT_LowDim_CrossFitting_Simulation.py --n_sample {2000,...,20000} --tag sizesweep` | 7 |
| Synthetic CATE (size sweep) | `Synthetic_CATE_CrossFitting_Simulation.py --n_sample {2000,...,20000} --head_batch 256 --tag hb256` | 6 |
| Kernel, Abortion | `DRKPV/Abortion_and_Crime_DRKPV_Simulation.py` | 1 |
| Kernel, Deaner IM / IR | `DRKPV/Deaner_DRKPV_Simulation.py [--deaner_id IR] --tag full` | 4 |
| Kernel, Demand | `DRKPV/Demand_DRKPV_Nystrom_Simulation.py --n_sample {2000,...,20000}` | 2 |
| Kernel, dSprite V2 | `DRKPV/dSprite_DRKPV_Nystrom_Simulation.py --n_sample {2000,...,20000}` | 9 |
| Kernel, dSprite V3 | `DRKPV/dSprite_DRKPV_Nystrom_Simulation.py --dgp v3 --n_sample {2000,...,20000}` | 11 |
| Kernel, synthetic ATT | `KPV/Synthetic_ATT_LowDim_KPV_KAP_Nystrom_Simulation.py --n_sample {2000,...,20000}` | 7 |
| Kernel, synthetic CATE | `KPV/Synthetic_CATE_KPV_Nystrom_Simulation.py --n_sample {2000,...,20000}` | 6 |

Bridge-route stress tests (Scenario B and Scenario O). Each cell is one invocation; the notebook reads one
folder per cell, so pass `--output_dir` accordingly:

```bash
cd Simulations/BridgeRoutePreference
# Scenario B, n = 4000, overlapping stage-1/stage-2 split
python scenario_b_treatment_bridge_degradation.py --kappa_s {0,0.2,0.4,0.6} --n_sample 4000 \
    --split overlap --data_seed <0-9> --train_seed <700000 + seed + 1000*repeat> \
    --output_dir ../Results/BridgeRoutePreference/scenarioB_n4000_overlap
# Scenario O, n = 4000, overlapping split, forward sampling of the latent
python scenario_o_compensated_proxy_reliability.py --lambda_ws {0.9,0.7,0.6,0.5,0.4} --n_sample 4000 \
    --split overlap --sampling forward --data_seed <0-9> --train_seed <800000 + seed + 1000*repeat> \
    --output_dir ../Results/BridgeRoutePreference/scenarioO_compensated_n4000_overlap_forward
```

Replace `overlap` by `independent` and `4000` by `2000` for the other three cells of each scenario; the
folder name must match (`scenarioB_n{2000,4000}_{overlap,independent}`,
`scenarioO_compensated_n{2000,4000}_{overlap,independent}_forward`). Notebook 12 reads them.

Proxy-informativeness ablation (n = 4000, `kkt` arm). One invocation per noise level, noising one proxy at
a time; `s = 0` is shared by both arms:

```bash
cd Simulations/DR-NME-PCL-Dens-Ratio-Free
python Demand_CrossFitting_Simulation.py --n_sample 4000 --normalization kkt --n_repeats 3 \
    --data_seeds 0 --z_noise 0.5 --tag pn_z0.5 --out_dir ../Results/Demand_ProxyNoise
python dSprite_V3_CrossFitting_Simulation.py --n_sample 4000 --arms kkt-mean --n_repeats 3 \
    --data_seeds 0 --w_noise 1 --tag pn_w1 --om_reg1_lo 1 --om_reg21_lo 1 --om_reg2_lo 10 --om_reg2_hi 10 \
    --om_trainer v2 --om_arch sn --y_scale raw --z_scale raw --om_batch 6000 --om_epochs 20 \
    --om_stage1_iter 20 --om_lr 1e-3 --om_weight_decay 0.1 --om_v2_bn_train_pred 1 --third_loss huber \
    --om_both 0 --out_dir ../Results/dSprite_V3_ProxyNoise
```

Levels: Demand `s ∈ {0, 0.25, 0.5, 1, 2, 4}`, dSprite V3 `s ∈ {0, 1, 4}`, with `--z_noise` and `--w_noise`
swept separately. The examples above show one data seed; the reported grid runs `--data_seeds 0 1 2 3 4 5 6 7
8 9` at every level, and notebook 13 reports the mean ± standard error over those 10 seeds without checking
coverage, so a partial grid gives a table that looks complete. Notebook 13 reads `Results/Demand_ProxyNoise`
and `Results/dSprite_V3_ProxyNoise`.

The size sweeps are array jobs in practice: one task per (sample size, data seed) cell, three repeats inside
each task. Any scheduler will do — the scripts take a single `--n_sample` and a single `--data_seeds` value
and need no other coordination.

## Analysis notebooks

`Simulations/AnalyzeSimulationResults/*.ipynb` load the CSVs from `Simulations/Results/`, print the tables
(causal MSE, mean ± standard error over the data seeds, with repeats averaged within a seed first) and draw
the size-sweep figures. They ship with the outputs they produced, so the tables and figures can be
read without re-running anything. The numbering follows the order the experiments appear in the paper,
so it is not consecutive. Notebooks 2, 6, 7, 9, 11 and 13 draw figures; those for Demand,
dSprite V2/V3, ATT and CATE are saved in `Figures/` (the two dSprite cells also save a linear-axis twin of
their log-axis figure). Notebooks 1, 4 and 12 report tables only.

## Baselines not included here

The NMMR and PKDR baselines were run with the authors' public code, unmodified, through thin wrappers that
call it on the same data seeds as our scripts. That code is not ours and is not redistributed here:

- NMMR: https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression
- PKDR: https://github.com/yezichu/PCL_Continuous_Treatment
- DFPV, and the demand and dSprites benchmark designs: https://github.com/liyuan9988/DeepFeatureProxyVariable

Notebooks 1, 2, 4, 9 and 11 show NMMR and PKDR rows beside ours. Those rows come from the wrappers described
above, which are not part of this repository, so they cannot be regenerated from it; each notebook prints `no
rows` for a family whose CSVs are absent and reports the remaining ones.

A notebook needs the CSVs of the grids it reports, so re-execute one only after running the corresponding
simulations; otherwise read the shipped outputs as they are. The dSprite scripts expose an optional
`--om_arch dfpv` that wraps the DFPV outcome bridge; it is off by default and expects the DFPV repository at
`../external/DeepFeatureProxyVariable`, a sibling of this repository.

The kernel baselines (KPV, KAP, DRKPV, and their Nyström variants) are our own implementations, in
`src/causal_learning/`.

## Citation

See `CITATION.cff`. The arXiv entry will be added here once the preprint is posted.

## License

MIT — see `LICENSE`.
