# rd.py - small-scale Recurrent-Depth LM (arXiv 2502.05171), same architecture as the notebook's
# Checkpoint 1-3 cells, plus: KV-cache inference, the training objective, and a multi-hop program task.
import math, random, time, json, os, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, asdict


# ═════════════════════════════════════ config ═════════════════════════════════════
@dataclass
class Cfg:
    vocab: int = 48
    h: int = 256
    n_heads: int = 4
    mlp_inner: int = 864          # 17920/5280 = 3.39 x h, as in the paper
    l_P: int = 1
    l_R: int = 2
    l_C: int = 1
    r_bar: int = 16               # mean recurrence (training); sets init depth l
    rope_base: float = 50000.0
    norm_eps: float = 1e-6
    max_seq: int = 512
    @property
    def head_dim(self): return self.h // self.n_heads
    @property
    def l_eff(self): return self.l_P + self.r_bar * self.l_R + self.l_C
    @property
    def std_in(self): return math.sqrt(2 / (5 * self.h))
    @property
    def std_out(self): return math.sqrt(1 / (5 * self.h * self.l_eff))
    @property
    def emb_scale(self): return math.sqrt(self.h)
    @property
    def std_s(self): return math.sqrt(2 / 5)


# ═════════════════════════════════════ layers ═════════════════════════════════════
def tnormal_(w, std, gen=None):
    return nn.init.trunc_normal_(w, 0.0, std, -3 * std, 3 * std, generator=gen)

class RMSNorm(nn.Module):
    def __init__(s, d, eps):
        super().__init__(); s.eps = eps; s.weight = nn.Parameter(torch.ones(d))
    def forward(s, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)).to(x.dtype) * s.weight.to(x.dtype)

def apply_rope(x, cos, sin):                 # x (B,H,T,d); cos/sin (T, d/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)

class KVSlot:
    """Preallocated K/V for one attention layer (one recurrence slot for core layers)."""
    def __init__(s, B, H, T, d, device, dtype):
        s.K = torch.zeros(B, H, T, d, device=device, dtype=dtype)
        s.V = torch.zeros(B, H, T, d, device=device, dtype=dtype)

class Attention(nn.Module):
    def __init__(s, cfg):
        super().__init__(); s.cfg = cfg; h = cfg.h
        s.q = nn.Linear(h, h, bias=True); s.k = nn.Linear(h, h, bias=True)     # bias on q,k only
        s.v = nn.Linear(h, h, bias=False); s.o = nn.Linear(h, h, bias=False)
    def forward(s, x, cos, sin, slot=None, p0=0):
        B, T, h = x.shape; H, d = s.cfg.n_heads, s.cfg.head_dim
        q, k, v = (m(x).view(B, T, H, d).transpose(1, 2) for m in (s.q, s.k, s.v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if slot is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                                  # write this chunk at [p0, p0+T), attend to [0, p0+T)
            slot.K[:, :, p0:p0 + T] = k.to(slot.K.dtype); slot.V[:, :, p0:p0 + T] = v.to(slot.V.dtype)
            K, V = slot.K[:, :, :p0 + T].to(q.dtype), slot.V[:, :, :p0 + T].to(q.dtype)
            mask = None
            if T > 1:
                mask = torch.arange(p0 + T, device=x.device)[None, :] <= (p0 + torch.arange(T, device=x.device))[:, None]
            y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        return s.o(y.transpose(1, 2).reshape(B, T, h))

class GatedMLP(nn.Module):
    def __init__(s, cfg):
        super().__init__()
        s.gate = nn.Linear(cfg.h, cfg.mlp_inner, bias=False); s.up = nn.Linear(cfg.h, cfg.mlp_inner, bias=False)
        s.down = nn.Linear(cfg.mlp_inner, cfg.h, bias=False)
    def forward(s, x): return s.down(F.silu(s.gate(x)) * s.up(x))

class SandwichLayer(nn.Module):
    """x_hat = n2(x + Attn(n1(x)));  x_out = n4(x_hat + MLP(n3(x_hat)))"""
    def __init__(s, cfg):
        super().__init__()
        s.n1, s.n2, s.n3, s.n4 = (RMSNorm(cfg.h, cfg.norm_eps) for _ in range(4))
        s.attn, s.mlp = Attention(cfg), GatedMLP(cfg)
    def forward(s, x, cos, sin, slot=None, p0=0):
        x_hat = s.n2(x + s.attn(s.n1(x), cos, sin, slot, p0))
        return s.n4(x_hat + s.mlp(s.n3(x_hat)))


# ═════════════════════════════════════ model ═════════════════════════════════════
class Cache:
    """KV cache. Keys: ('P',l), ('C',l), ('R',l,it). With budget b, core slot it -> it % b (Sec 6.2)."""
    def __init__(s, model, B, T, device, dtype=torch.float32, budget=None):
        s.cfg = model.cfg; s.B, s.T, s.device, s.dtype, s.budget = B, T, device, dtype, budget; s.slots = {}
    def key(s, k):
        return ("R", k[1], k[2] % s.budget) if (k[0] == "R" and s.budget) else k
    def slot(s, k):
        k = s.key(k)
        if k not in s.slots:
            s.slots[k] = KVSlot(s.B, s.cfg.n_heads, s.T, s.cfg.head_dim, s.device, s.dtype)
        return s.slots[k]
    def copy_core(s, it_from, it_to, pos, rows):
        """For batch rows `rows` (bool), copy core K/V at positions `pos` from iteration it_from to it_to
        ('attend to the last, deepest available KV state', Remark 6.1)."""
        for l in range(s.cfg.l_R):
            a, b = s.slot(("R", l, it_from)), s.slot(("R", l, it_to))
            if a is b: continue
            b.K[rows, :, pos] = a.K[rows, :, pos]; b.V[rows, :, pos] = a.V[rows, :, pos]

class RecurrentDepthLM(nn.Module):
    def __init__(s, cfg, seed=0):
        super().__init__(); s.cfg = cfg; h = cfg.h
        s.E = nn.Embedding(cfg.vocab, h)
        s.prelude = nn.ModuleList(SandwichLayer(cfg) for _ in range(cfg.l_P))
        s.A = nn.Linear(2 * h, h, bias=False)                                 # adapter [s; e] -> h
        s.core = nn.ModuleList(SandwichLayer(cfg) for _ in range(cfg.l_R))
        s.n_R = RMSNorm(h, cfg.norm_eps)                                      # n_c of the core
        s.coda = nn.ModuleList(SandwichLayer(cfg) for _ in range(cfg.l_C))
        s.n_C = RMSNorm(h, cfg.norm_eps)                                      # final norm before tied head
        inv = 1.0 / (cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        ang = torch.outer(torch.arange(cfg.max_seq).float(), inv)
        s.register_buffer("cos", ang.cos(), persistent=False); s.register_buffer("sin", ang.sin(), persistent=False)
        s.reset_parameters(seed)

    @torch.no_grad()
    def reset_parameters(s, seed=0):
        g = torch.Generator().manual_seed(seed)
        for name, p in s.named_parameters():
            if p.dim() == 1:
                p.fill_(1.0) if name.endswith("weight") else p.zero_()
            else:
                w = torch.empty(p.shape)
                tnormal_(w, s.cfg.std_out if name.endswith(("attn.o.weight", "mlp.down.weight")) else s.cfg.std_in, g)
                p.copy_(w)

    def tables(s, p0, T): return s.cos[p0:p0 + T], s.sin[p0:p0 + T]

    def embed(s, ids, p0=0, cache=None):                                       # e = P(x)
        cos, sin = s.tables(p0, ids.shape[1])
        x = s.E(ids) * s.cfg.emb_scale
        for l, L in enumerate(s.prelude):
            x = L(x, cos, sin, cache.slot(("P", l)) if cache else None, p0)
        return x

    def core_step(s, e, st, p0=0, cache=None, it=0):                          # s_i = R(e, s_{i-1})
        cos, sin = s.tables(p0, e.shape[1])
        x = s.A(torch.cat([st.to(e.dtype), e], -1))
        for l, L in enumerate(s.core):
            x = L(x, cos, sin, cache.slot(("R", l, it)) if cache else None, p0)
        return s.n_R(x)

    def decode(s, st, p0=0, cache=None, return_hidden=False):               # p = C(s)
        cos, sin = s.tables(p0, st.shape[1])
        x = st
        for l, L in enumerate(s.coda):
            x = L(x, cos, sin, cache.slot(("C", l)) if cache else None, p0)
        x = s.n_C(x)
        logits = x @ s.E.weight.T.to(x.dtype)
        return (logits, x) if return_hidden else logits

    def init_state(s, shape, device, gen=None, kind="random"):
        if kind == "zeros": return torch.zeros(shape, device=device)
        w = torch.empty(shape, device=device)
        return tnormal_(w, s.cfg.std_s, gen)

    def forward(s, ids, r, k=None, s0=None, gen=None, return_states=False):
        """Full-sequence forward. r iterations; if k is given, gradients flow only through the last k
        (truncated backprop, Sec 3.3). return_states -> list [s_0..s_r] (detached)."""
        e = s.embed(ids)
        st = s.init_state(e.shape, e.device, gen) if s0 is None else s0
        states = [st.detach()] if return_states else None
        n_free = 0 if k is None else max(0, r - k)
        with torch.no_grad():
            for i in range(n_free):
                st = s.core_step(e, st)
                if return_states: states.append(st.detach())
        if n_free:
            # autocast caches the fp16 weight casts it made under no_grad; those copies have no grad_fn, and
            # reusing them would silently cut every core weight out of the gradient. Drop the cache.
            torch.clear_autocast_cache()
        for i in range(n_free, r):
            st = s.core_step(e, st)
            if return_states: states.append(st.detach())
        logits = s.decode(st)
        return (logits, states) if return_states else logits


# ═══════════════════════ training objective: r ~ log-normal Poisson ═══════════════════════
def sample_r(r_bar, sigma=0.5, rng=None):
    """Eq. (1)-(2): tau ~ N(log r_bar - sigma^2/2, sigma),  r ~ Poisson(e^tau) + 1.  E[r] = r_bar + 1."""
    rng = rng or random
    tau = rng.gauss(math.log(r_bar) - 0.5 * sigma ** 2, sigma)
    lam = math.exp(tau)
    # Poisson sample by inversion (exact, lam is O(10-100))
    u, k, p = rng.random(), 0, math.exp(-lam)
    c = p
    while u > c and k < 10000:
        k += 1; p *= lam / k; c += p
    return k + 1


# ═══════════════════════ task: multi-hop programs (depth = # dependent ops) ═══════════════════════
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SYMS = list(LETTERS) + list("0123456789") + list("+-*=;?") + ["^", "$"]      # ^ BOS, $ EOS
STOI = {c: i for i, c in enumerate(SYMS)}
BOS, EOS = STOI["^"], STOI["$"]
OPS = {"+": lambda a, b: (a + b) % 10, "-": lambda a, b: (a - b) % 10, "*": lambda a, b: (a * b) % 10}

def make_program(n_vars, n_q, rng, chains=(2, 4), d_max=2):
    """Task v5. Statements 'N=O+d;' (5 chars + ';'), d in {0,1}, values mod 10. The variables form 2-4 independent
    chains of random lengths whose statements are interleaved at random (each chain keeps its order): a chain
    starts with a digit root and every later link is 'previous link + d'. Depth = position in its chain.
    (v4 grew one dominant chain, so a variable's chain was ~ the whole prefix and 'root + #(+1) so far' was an
    O(1)-layer shortcut: the r=1 twin scored 46% at depth 12. Interleaving breaks that.) -> text, answers, depths"""
    names = rng.sample(LETTERS, n_vars)
    c = min(rng.randint(*chains), n_vars)
    cuts = sorted(rng.sample(range(1, n_vars), c - 1))
    lens = [b - a for a, b in zip([0] + cuts, cuts + [n_vars])]
    order = [ci for ci, l in enumerate(lens) for _ in range(l)]; rng.shuffle(order)
    last = [None] * c; val, dep, stmts = {}, {}, []
    for j, ci in enumerate(order):
        nm, d = names[j], rng.randrange(d_max)
        if last[ci] is None:
            a = rng.randrange(10); src = str(a); val[nm] = (a + d) % 10; dep[nm] = 1
        else:
            par = last[ci]; src = par; val[nm] = (val[par] + d) % 10; dep[nm] = dep[par] + 1
        last[ci] = nm; stmts.append(f"{nm}={src}+{d};")
    qs = rng.sample(names, n_q)
    text = "".join(stmts) + "?" + "".join(q + str(val[q]) for q in qs) + "$"      # '?k3q7a1$': value right after its name
    return text, [val[q] for q in qs], [dep[q] for q in qs]

def make_batch(B, n_vars, n_q, rng, device="cpu"):
    rows, depths = [], []
    for _ in range(B):
        t, a, d = make_program(n_vars, n_q, rng)
        rows.append([BOS] + [STOI[c] for c in t]); depths.append(d)
    ids = torch.tensor(rows, device=device)
    L = ids.shape[1]; ans_pos = torch.arange(L - 2 * n_q, L - 1, 2, device=device)  # the n_q value digits
    return ids, ans_pos, torch.tensor(depths, device=device)

def answer_loss_acc(logits, ids, ans_pos):
    """Next-token CE restricted to the answer digits (predicted from the token before each)."""
    lg = logits[:, ans_pos - 1].float(); tg = ids[:, ans_pos]
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1))
    return loss, (lg.argmax(-1) == tg)

def hidden_corr(st):
    """Mean cosine between different positions of the latent state (Fig 5 collapse metric), BOS excluded."""
    x = F.normalize(st[:, 1:].float(), dim=-1); n = x.shape[1]
    c = x @ x.transpose(1, 2)
    return ((c.sum((1, 2)) - n) / (n * (n - 1))).mean().item()


# ═══════════════════════ inference: cached generation + Sec 6 features ═══════════════════════
@torch.no_grad()
def generate(model, prompt, n_new, r, exit_kl=None, budget=None, warm_start=False, s0_kind="random",
             seed=0, return_logits=False, force=None):
    """Greedy decoding with a per-iteration KV cache.
    prompt[:, :-1] is prefilled with r iterations; then n_new tokens are decoded, starting from prompt[:, -1].
    exit_kl  : per-token early exit when KL(p_i || p_{i-1}) < exit_kl (Sec 6.1); exited tokens expose their
               last KV to later iterations (Remark 6.1).
    budget   : recurrent KV-cache budget, slot = i mod budget (Sec 6.2).
    warm_start: s0 of each new token = final state of the previous token (Sec 6.3, continuous CoT).
    force    : optional (B, n_new) ids; entries >= 0 are emitted instead of the model's choice (teacher-forced
               query names), entries < 0 are free.
    Returns tokens (B,n_new), steps used (B,n_new) [, logits (B,n_new,V)]."""
    dev = prompt.device; B, n = prompt.shape
    cache = Cache(model, B, n + n_new, dev, budget=budget)
    g = torch.Generator(device=dev).manual_seed(seed)
    e = model.embed(prompt[:, :-1], 0, cache)
    st = model.init_state(e.shape, dev, g, s0_kind)
    for i in range(r):
        st = model.core_step(e, st, 0, cache, i)
    model.decode(st, 0, cache)
    prev = st[:, -1:]
    tok = prompt[:, -1:]; toks, steps, logs = [], [], []
    for t in range(n_new):
        p0 = n - 1 + t
        e = model.embed(tok, p0, cache)
        st = prev.clone() if warm_start else model.init_state(e.shape, dev, g, s0_kind)
        active = torch.ones(B, dtype=torch.bool, device=dev)
        used = torch.full((B,), r, device=dev)
        lp_prev = None
        for i in range(r):
            new = model.core_step(e, st, p0, cache, i)
            st = torch.where(active[:, None, None], new, st)
            if i > 0 and not active.all():
                cache.copy_core(i - 1, i, p0, ~active)
            if exit_kl is not None:
                lp = model.decode(st, p0, cache).float().log_softmax(-1)[:, 0]
                if lp_prev is not None:
                    kl = (lp.exp() * (lp - lp_prev)).sum(-1)
                    newly = active & (kl < exit_kl)
                    used[newly] = i + 1; active = active & ~newly
                lp_prev = lp
                if not active.any():
                    for j in range(i + 1, r):
                        cache.copy_core(j - 1, j, p0, torch.ones(B, dtype=torch.bool, device=dev))
                    break
        logits = model.decode(st, p0, cache)[:, 0]
        tok = logits.argmax(-1, keepdim=True)
        if force is not None:
            tok = torch.where(force[:, t:t + 1] >= 0, force[:, t:t + 1], tok)
        toks.append(tok); steps.append(used); logs.append(logits); prev = st
    out = (torch.cat(toks, 1), torch.stack(steps, 1))
    return out + (torch.stack(logs, 1),) if return_logits else out

@torch.no_grad()
def speculative_one(model, prompt, n_new, r_draft, r_verify, seed=0, force=None):
    """Self-speculative decoding (Sec 6.4) for one sequence: draft the remaining tokens with r_draft,
    verify all of them in one parallel pass at r_verify, keep the agreeing prefix + the verifier's next token.
    Returns tokens (1,n_new), sequential core iterations used, drafted, accepted."""
    ctx = prompt; done = 0; cost = 0; drafted = 0; accepted = 0; rnd = 0
    while done < n_new:
        m = n_new - done
        fz = None if force is None else force[:, done:]
        d, _ = generate(model, ctx, m, r_draft, seed=seed + rnd, force=fz)
        ver = model(torch.cat([ctx, d[:, :-1]], 1), r_verify,
                    gen=torch.Generator(device=ctx.device).manual_seed(seed + 1000 + rnd))[:, ctx.shape[1] - 1:].argmax(-1)
        if fz is not None: ver = torch.where(fz >= 0, fz, ver)
        cost += m * r_draft + r_verify; drafted += m
        agree = (ver[0] == d[0]).long()
        a = int(agree.cumprod(0).sum().item())
        take = min(a + 1, m); accepted += min(a, m)
        ctx = torch.cat([ctx, ver[:, :take]], 1); done += take; rnd += 1
    return ctx[:, prompt.shape[1]:], cost, drafted, accepted


# ═════════════════════════════════════ training ═════════════════════════════════════
def train(out, gpu=0, steps=6000, bs=128, lr=5e-4, r_bar=16, k=8, fixed_r=None, n_vars=12, n_q=0,
          seed=0, eval_every=250, warmup=300, cfg_over=None, n_vars_min=None, init=None):
    """n_q=0 -> query every variable. n_vars_min -> each batch draws its program size from [n_vars_min, n_vars]."""
    torch.manual_seed(seed); rng = random.Random(seed)
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    cfg = Cfg(r_bar=r_bar, **(cfg_over or {}))
    model = RecurrentDepthLM(cfg, seed=seed).to(dev)
    if init: model.load_state_dict(torch.load(init, map_location=dev)["model"])      # continue from a checkpoint
    decay = [p for n_, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n_, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warmup))   # warmup, then constant
    use_amp = dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    vrng = random.Random(12345)
    val = make_batch(512, n_vars, n_q or n_vars, vrng, dev)
    os.makedirs(out, exist_ok=True)
    json.dump({"cfg": asdict(cfg), "args": dict(steps=steps, bs=bs, lr=lr, r_bar=r_bar, k=k, fixed_r=fixed_r,
               n_vars=n_vars, n_q=n_q, seed=seed, n_vars_min=n_vars_min)}, open(f"{out}/config.json", "w"))
    log = open(f"{out}/log.jsonl", "a"); t0 = time.time(); run_loss = None

    def evaluate(step):
        model.eval(); rec = {"step": step, "time": time.time() - t0}
        ids, ap, dp = val
        g = torch.Generator(device=dev).manual_seed(0)
        for r in ([1] if fixed_r == 1 else [1, 2, 4, 8, 16, 32]):
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=use_amp):
                lg, states = model(ids, r, gen=g, return_states=True)
            loss, ok = answer_loss_acc(lg, ids, ap)
            rec[f"acc_r{r}"] = ok.float().mean().item(); rec[f"loss_r{r}"] = loss.item()
            rec[f"corr_r{r}"] = hidden_corr(states[-1])
        rec["train_loss"] = run_loss
        log.write(json.dumps(rec) + "\n"); log.flush(); model.train()
        print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rec.items()}), flush=True)

    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0: evaluate(step)
        if step == steps: break
        nv = rng.randint(n_vars_min or n_vars, n_vars)
        ids, ap, _ = make_batch(bs, nv, n_q or nv, rng, dev)
        r = fixed_r if fixed_r else sample_r(r_bar, rng=rng)
        with torch.autocast("cuda", torch.float16, enabled=use_amp):
            lg = model(ids, r, k=k)
            loss, _ = answer_loss_acc(lg, ids, ap)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if (step + 1) % 1000 == 0:
            torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "step": step + 1}, f"{out}/ckpt.pt")
    torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "step": steps}, f"{out}/ckpt.pt")
    open(f"{out}/DONE", "w").write("ok")

def load(path, device):
    ck = torch.load(path, map_location=device)
    m = RecurrentDepthLM(Cfg(**ck["cfg"])).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--r_bar", type=int, default=16)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--fixed_r", type=int, default=None)
    ap.add_argument("--n_vars", type=int, default=12); ap.add_argument("--n_q", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_vars_min", type=int, default=None); ap.add_argument("--init", default=None)
    a = ap.parse_args()
    train(a.out, a.gpu, a.steps, a.bs, a.lr, a.r_bar, a.k, a.fixed_r, a.n_vars, a.n_q, a.seed, a.eval_every,
          n_vars_min=a.n_vars_min, init=a.init)
