# Cluster jobs

The SLURM scripts that produced the thesis data, one per stage (see the stage table in the root
README). They are the record of how every job was run: which controllers were trained with which
settings, which scenario lists were evaluated, and which checks were made.

The scripts were adapted to the layout of this repository (module names and paths) and checked
for syntax, but they were not submitted again in this form. The jobs themselves ran from the
original working tree; their logs are in `data/logs/` and in `data/results/sealed/` and
`data/results/stage1/`.

Submit from the repository root, on one node, with a virtual environment in `.venv`:

```bash
sbatch --nodelist=<node> cluster/stage4_seeds_83_149.sh
```

- Every script starts with the parity gate and stops if it fails.
- A step is skipped when its output already exists. With the default `OUT=data/results/stage<N>`
  everything is found and nothing runs. Set `OUT=runs/results/stage<N>` to evaluate again, and
  the checkpoint variable of the script (`MODELS`, `NEW` or `OUTPUT`) to `runs/checkpoints` to
  train again.
- `#SBATCH` lines (partition, constraint, node names) belong to the cluster the study used and
  need to be changed for another one.
- Job names (`thesis-plan2`, `thesis-posthoc-s1`, ...) are the original ones, so new logs match
  the stored ones.
