# Recurrent Depth, built step by step

A from-scratch PyTorch build of **"Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"** (Geiping et al., arXiv 2502.05171), done one checkpoint at a time in a live Colab/Kaggle notebook on 2x Tesla T4.

Only the paper was used as the source. The single external artifact is the paper's tokenizer (`tokenizer.json` of `tomg-group-umd/huginn-0125`). No model code or weights were taken from anywhere.

```
e = P(x)                        prelude: embed + 2 sandwich layers
s_0 ~ TruncNormal(0, 2/5)
s_i = R(e, s_{i-1}),  i=1..r    core: adapter A[s; e] + 4 sandwich layers + RMSNorm, looped
p = C(s_r)                      coda: 2 sandwich layers + RMSNorm + tied head
```

## What is in here

| path | what |
|---|---|
| `recurrent_depth_stepwise.ipynb` | the whole build, Checkpoints 1 to 7, **with outputs** |
| `rd.py` | the small-scale library: model, KV cache, Sec 6 inference features, r sampler, task, training loop. Final version (md5 `00f5408...`) |
| `runs/<name>/ckpt.pt` | trained checkpoints (see table below) |
| `runs/<name>/log.jsonl` | eval log every 250 steps: acc/loss/hidden-state corr at r = 1,2,4,8,16,32 |
| `runs/_v*` , `runs/_invalid_*` | logs of the abandoned task versions and the run hit by the autocast bug, kept as evidence |
| `figures/` | the notebook's figures at full resolution |
| `tests/test_rd.py` | CPU tests: cached decode == full forward, KV budget, exit, truncated-BPTT grads, r sampler |
| `MANIFEST.json` | size + md5 of every exported file |

### Checkpoints

| run | steps | what | continue from |
|---|---|---|---|
| `recurrent` | 0 to 6000 | r ~ log-normal Poisson(r̄=16), truncated backprop k=8, seed 0 | `runs/recurrent/ckpt.pt` |
| `recurrent_c` | 6000 to 12000 | continuation of `recurrent` (**final, seed 0**) | `runs/recurrent_c/ckpt.pt` |
| `recurrent_s1`, `recurrent_s1_c` | same, seed 1 | replication | `runs/recurrent_s1_c/ckpt.pt` |
| `twin_r1`, `twin_r1_c` | 0 to 12000 | non-recurrent twin: same network trained with r = 1 | `runs/twin_r1_c/ckpt.pt` |

Model: h=256, 4 heads, MLP 864, shape (l_P, l_R, l_C) = (1, 2, 1), 3.9M parameters, 15 MB per checkpoint. A checkpoint stores `cfg` + `model` state dict + `step` (counted within that run, so a `_c` checkpoint says 6000 but is 12000 total). **The optimizer state is not saved**, so a resume restarts AdamW moments (with a 300-step warm-up).

## How to continue from here

**1. Get a GPU runtime** (Kaggle 2xT4 or Colab), then:
```bash
git clone https://github.com/Maverick-Ansh/recurrent-depth-stepwise
cd recurrent-depth-stepwise
```

**2. Load a trained model and evaluate it**
```python
import sys; sys.path.insert(0, "."); import rd, random, torch
m = rd.load("runs/recurrent_c/ckpt.pt", "cuda")
ids, ans_pos, depth = rd.make_batch(512, 12, 12, random.Random(1000), "cuda")
with torch.no_grad(), torch.autocast("cuda", torch.float16):
    for r in (1, 4, 16, 64):
        _, ok = rd.answer_loss_acc(m(ids, r), ids, ans_pos); print(r, ok.float().mean().item())
```
The notebook's TEST set is `depth_batch(512, 1000 + i)` for i = 0..3 and DEEP is `chains=(2, 2)` with seeds 2000 + i. Both are deterministic, so no data needs saving.

**3. Train further** (one process per GPU, logs to `<out>/log.jsonl`, checkpoint every 1000 steps):
```bash
python rd.py --out runs/recurrent_c2 --gpu 0 --r_bar 16 --k 8 --steps 6000 --lr 1e-3 --n_vars_min 4 \
             --init runs/recurrent_c/ckpt.pt --seed 20
python rd.py --out runs/twin_c2      --gpu 1 --fixed_r 1          --steps 6000 --lr 1e-3 --n_vars_min 4 \
             --init runs/twin_r1_c/ckpt.pt --seed 20
```
A fresh run is the same command without `--init`. The main knobs are `--r_bar`, `--k`, `--fixed_r`, `--n_vars`, `--n_vars_min`, `--n_q` (0 = query every variable), `--lr`, `--bs`. The model shape is the `Cfg` dataclass at the top of `rd.py` (pass `cfg_over=dict(h=512, ...)` to `rd.train` from Python).

**4. Re-run the notebook.** Checkpoints 1 to 3 rebuild the **paper-scale** 3.5B model from fixed seeds (needs about 10.5 GB on GPU 0 and 9 GB on GPU 1, fp32). They produce the same numbers every time, so nothing from them was saved. From Checkpoint 4 on, the notebook writes `rd.py` with a `%%writefile` cell (an older version) and then applies the task-v4, task-v5 and `--init` patches in the cells after it. The result is byte-identical to the `rd.py` in this repo. When resuming, **skip those cells and use the repo's `rd.py`**.

**5. Run the tests** (CPU, about 2 min): `python tests/test_rd.py`

## Results in one table

Checkpoints 1 to 4 were done at the paper's full scale (untrained). Checkpoints 5 to 7 use the small model trained for 12k steps on the interleaved-chain task (chance 0.104).

| paper claim | result here |
|---|---|
| 3.5B params, init, sandwich norms | 3.565B, initial loss 11.24 (predicted 11.28) |
| Fig 3 r sampler: mean 33, median 29, mode 24 | 32.95 / 29 / 24 |
| truncated backprop keeps memory flat in r | 1.48 GB at r = 16..64 (full backprop 3.3 to 12.5 GB) |
| footnote 1: recurrence unstable without injecting e | untrained contraction per step = 1/sqrt(1+α²), measured to 3 decimals, 1.000 at α = 0 |
| more r helps, harder items saturate later (Fig 7) | yes, both seeds: saturation r grows 1 to 6 with depth. No gain beyond r ≈ 6 |
| recurrent beats its non-recurrent twin (Table 4) | +1 point overall, +6 points at depth >= 8 |
| zero-shot KV-cache sharing | no loss at any budget 1..32 |
| self-speculative decoding | 3.4x fewer sequential core iterations, same accuracy |
| per-token adaptive exit follows difficulty (Fig 10) | exit saves 8x compute vs r=64 with no loss, but steps barely depend on depth |
| continuous CoT saves 1-2 steps | no saving |
| path independence | holds for 97.5% of tokens (median seed gap 2e-7) |

**Caveats that matter if you continue:**
- The models are weak in absolute terms: depth-1 accuracy is only about 0.45, so the base look-up is the bottleneck. Training longer or wider is the obvious next step.
- **Answer leakage:** querying all 12 variables lets later queries copy earlier revealed values under teacher forcing (accuracy by query index rises 0.27 to 0.51). Use `--n_q` < `n_vars`, or score only the first query.

## Bugs and traps found along the way
- **fp16 autocast + `no_grad` truncated backprop zeroes every core-weight gradient.** Autocast caches the fp16 cast of each weight on first use, and inside `no_grad` that cached copy has no grad_fn, so the last k iterations reuse it. The fix is `torch.clear_autocast_cache()` after the no-grad loop (in `rd.py`). CPU tests cannot see this.
- **Task shortcuts.** One-dominant-chain programs let a 4-layer model count its way to deep answers (the twin scored 0.46 at depth 12). Interleaving independent chains fixed that.
- `AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")` blocks forever on a hidden trust_remote_code prompt. Load `tokenizer.json` with `PreTrainedTokenizerFast` instead.
- `copy.deepcopy(model).to("cuda:1")` clones on the source GPU first and can run it out of memory.
