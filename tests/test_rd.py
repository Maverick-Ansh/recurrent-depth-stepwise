import random, statistics, math, torch, collections
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); import rd
torch.manual_seed(0)
cfg = rd.Cfg(h=64, n_heads=2, mlp_inner=128, r_bar=4)
m = rd.RecurrentDepthLM(cfg, seed=1).eval()
# make the model non-trivial (paper init is near-identity; perturb out-projections so bugs show up)
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.dim() == 2: p.add_(torch.randn_like(p) * 0.05)

rng = random.Random(0)
ids, ap, dp = rd.make_batch(3, 6, 3, rng)
print("seq len", ids.shape, "example:", "".join(rd.SYMS[i] for i in ids[0].tolist()))

# 1) cached decode == full forward (zeros s0 so both are deterministic)
r = 5
prompt = ids[:, :-4]; n_new = 3
toks, steps, logits_c = rd.generate(m, prompt, n_new, r, s0_kind="zeros", return_logits=True)
full_in = torch.cat([prompt, toks[:, :-1]], 1)
lg_full = m(full_in, r, s0=torch.zeros(full_in.shape[0], full_in.shape[1], cfg.h))
lg_full = lg_full[:, prompt.shape[1] - 1:]
print("1) cached vs full max|dlogit|:", (lg_full - logits_c).abs().max().item())

# 2) KV budget >= r is identical; budget < r differs
_, _, lb = rd.generate(m, prompt, n_new, r, s0_kind="zeros", budget=r, return_logits=True)
_, _, l2 = rd.generate(m, prompt, n_new, r, s0_kind="zeros", budget=2, return_logits=True)
print("2) budget=r diff:", (lb - logits_c).abs().max().item(), "| budget=2 diff:", (l2 - logits_c).abs().max().item())

# 3) exit_kl=0 (never exits) identical; exit_kl large exits at step 2
_, s0, l0 = rd.generate(m, prompt, n_new, r, s0_kind="zeros", exit_kl=0.0, return_logits=True)
_, sH, _ = rd.generate(m, prompt, n_new, r, s0_kind="zeros", exit_kl=1e9, return_logits=True)
print("3) exit_kl=0 diff:", (l0 - logits_c).abs().max().item(), "steps", s0.tolist()[0], "| exit huge steps", sH.tolist()[0])

# 3b) exit semantics: token exiting at step j must equal a run where later KV of that token = its step-j KV.
_, s_mid, l_mid = rd.generate(m, prompt, n_new, r, s0_kind="zeros", exit_kl=1e-3, return_logits=True)
print("3b) moderate exit steps:", s_mid.tolist())

# 4) truncated backprop: grads with k equal grads of a manual detach version
m.train()
ids2, ap2, _ = rd.make_batch(4, 6, 3, rng)
s0t = m.init_state((4, ids2.shape[1], cfg.h), "cpu", torch.Generator().manual_seed(3))
def manual(k, r):
    e = m.embed(ids2); st = s0t
    for i in range(r):
        if i == r - k: st = st.detach()
        st = m.core_step(e if i >= r - k else e.detach(), st)
    return m.decode(st)
for (kk, rr) in [(2, 6), (8, 3)]:
    m.zero_grad(); l1, _ = rd.answer_loss_acc(m(ids2, rr, k=kk, s0=s0t), ids2, ap2); l1.backward()
    g1 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(); l2_, _ = rd.answer_loss_acc(manual(min(kk, rr), rr), ids2, ap2); l2_.backward()
    g2 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    print(f"4) k={kk} r={rr}: loss diff {abs(l1.item()-l2_.item()):.2e}, max grad diff",
          max((g1[n] - g2[n]).abs().max().item() for n in g1), "| params with grad", len(g1), "/", len(list(m.parameters())))
m.eval()

# 5) r sampler: paper Fig 3 (r_bar=32): mean 33.0, median 29, mode 24
rr = [rd.sample_r(32, rng=random.Random(i)) for i in range(200000)]
mode = collections.Counter(rr).most_common(1)[0][0]
print(f"5) r sampler r_bar=32: mean {statistics.mean(rr):.2f} median {statistics.median(rr)} mode {mode}  (paper 33.0/29.0/24.0)")

# 6) depth distribution of the task
rng = random.Random(1); dd = collections.Counter()
for _ in range(3000):
    _, _, d = rd.make_program(12, 6, rng); dd.update(d)
print("6) query depth histogram (n_vars=12):", dict(sorted(dd.items())))

# 7) speculative decoding runs and returns n_new tokens
out, cost, dr, ac = rd.speculative_one(m, prompt[:1], 3, 2, 5)
print("7) speculative:", out.tolist(), "cost", cost, "drafted", dr, "accepted", ac)
