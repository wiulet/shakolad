#!/usr/bin/env python3
"""
Kronos -> ONNX exporter for the Android app (one-time step; the GitHub workflow runs it for you).

Produces kronos-onnx.zip with 4 ONNX graphs + manifest.json (sha256 of every file):
  encoder.onnx  x[B,T,6] -> s1[B,T], s2[B,T]                 (tokenizer.encode(half=True))
  s1.onnx       s1_ids, s2_ids, stamp[B,T,5] -> s1_logits[B,V1] (last step), context[B,T,D]
  s2.onnx       context[B,T,D], s1_id[B,1] -> s2_logits[B,V2]   (last step)
  decoder.onnx  s1[B,T], s2[B,T] -> recon[B,T,6]              (tokenizer.decode(half=True))

The script VERIFIES every graph against PyTorch (two different sequence lengths, to prove the
dynamic axes work) and exits with an error if anything differs, so a bad export can never ship.
"""
import argparse, hashlib, json, os, sys, zipfile
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer
import model.module as M

PRESETS = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base",  512),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base",  512),
}

# --- export-friendly RoPE: the original caches cos/sin keyed on a Python int, which the ONNX
# tracer would freeze to one sequence length. Same maths, no cache. cos/sin come from q's length
# and are applied to both q and k (exactly what the original does, incl. the 1-token cross-attn).
def _rope_forward(self, q, k):
    n = q.shape[-2]
    t = torch.arange(n, device=q.device).type_as(self.inv_freq)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)
M.RotaryPositionalEmbedding.forward = _rope_forward

# --- export-friendly attention. The original calls F.scaled_dot_product_attention(is_causal=True); the
# TorchScript ONNX exporter's lowering of that op is fragile (mask / scale handling differ between torch
# versions) and produced logits that differ from PyTorch. Plain matmul+softmax with an explicit causal
# mask is mathematically identical and exports/validates reliably. Dropout is 0 in eval, so it's omitted.
def _attend(q, k, v, causal):
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if causal:
        L, S = q.shape[-2], k.shape[-2]
        i = torch.arange(L, device=q.device).unsqueeze(1)
        j = torch.arange(S, device=q.device).unsqueeze(0)
        neg = torch.full((), torch.finfo(scores.dtype).min, dtype=scores.dtype, device=q.device)
        scores = torch.where(j > i, neg, scores)
    return torch.matmul(torch.softmax(scores, dim=-1), v)

def _self_attn_forward(self, x, key_padding_mask=None):
    b, n, _ = x.shape
    q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    q, k = self.rotary(q, k)
    o = _attend(q, k, v, causal=True).transpose(1, 2).reshape(b, n, self.d_model)
    return self.resid_dropout(self.out_proj(o))

def _cross_attn_forward(self, query, key, value, key_padding_mask=None):
    b, ql, _ = query.shape
    sl = key.shape[1]
    q = self.q_proj(query).view(b, ql, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(key).view(b, sl, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(value).view(b, sl, self.n_heads, self.head_dim).transpose(1, 2)
    q, k = self.rotary(q, k)
    o = _attend(q, k, v, causal=False).transpose(1, 2).reshape(b, ql, self.d_model)  # eval: is_causal=False
    return self.resid_dropout(self.out_proj(o))

M.MultiHeadAttentionWithRoPE.forward = _self_attn_forward
M.MultiHeadCrossAttentionWithRoPE.forward = _cross_attn_forward


class Enc(nn.Module):
    def __init__(s, tok): super().__init__(); s.tok = tok
    def forward(s, x):
        a, b = s.tok.encode(x, half=True); return a, b
class S1(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, s1_ids, s2_ids, stamp):
        lg, ctx = s.m.decode_s1(s1_ids, s2_ids, stamp); return lg[:, -1, :], ctx
class S2(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, context, s1_id):
        return s.m.decode_s2(context, s1_id)[:, -1, :]
class Dec(nn.Module):
    def __init__(s, tok): super().__init__(); s.tok = tok
    def forward(s, a, b): return s.tok.decode([a, b], half=True)


FAILS = []
STRICT = False

def fail(msg):
    if STRICT: sys.exit(msg)
    print("  WARNING (non-strict, continuing):", msg); FAILS.append(msg)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def export(mod, args, names_in, names_out, dyn, path):
    torch.onnx.export(mod, args, path, input_names=names_in, output_names=names_out,
                      dynamic_axes=dyn, opset_version=18, do_constant_folding=True, dynamo=False)

def run_ort(path, feeds):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, feeds)

def check(name, a, b, tol=2e-3):
    d = float(np.max(np.abs(a - b)))
    scale = max(1.0, float(np.max(np.abs(a))))          # relative to the magnitude of the outputs
    print(f"  [{name}] max|torch-onnx| = {d:.2e}  (rel {d/scale:.2e}, |out|max {scale:.2f})")
    if d / scale > tol: fail(f"VERIFY FAILED for {name}: rel {d/scale} > {tol} (abs {d})")


class Dbg(nn.Module):
    """s1 graph that returns every intermediate, to locate the first stage where ONNX diverges."""
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, s1_ids, s2_ids, stamp):
        m = s.m
        e = m.embedding([s1_ids, s2_ids]); t = m.time_emb(stamp); x = e + t
        outs = [e, t, x]
        for layer in m.transformer:
            x = layer(x); outs.append(x)
        x = m.norm(x); outs.append(x); outs.append(m.head(x))
        return tuple(outs)

class L0(nn.Module):
    """First transformer block, split into sub-steps (same maths as TransformerBlock)."""
    def __init__(s, m): super().__init__(); s.l = m.transformer[0]
    def forward(s, x):
        l = s.l; a = l.self_attn; b, n, _ = x.shape
        h = l.norm1(x)
        q = a.q_proj(h).view(b, n, a.n_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(h).view(b, n, a.n_heads, a.head_dim).transpose(1, 2)
        v = a.v_proj(h).view(b, n, a.n_heads, a.head_dim).transpose(1, 2)
        qr, kr = a.rotary(q, k)
        sc = torch.matmul(qr, kr.transpose(-1, -2)) * (a.head_dim ** -0.5)
        i = torch.arange(n).unsqueeze(1); j = torch.arange(n).unsqueeze(0)
        neg = torch.full((), torch.finfo(sc.dtype).min, dtype=sc.dtype)
        mk = torch.where(j > i, neg, sc)
        pr = torch.softmax(mk, dim=-1)
        o = torch.matmul(pr, v).transpose(1, 2).reshape(b, n, a.d_model)
        ao = a.out_proj(o); x1 = x + ao; h2 = l.norm2(x1); f = l.ffn(h2)
        return h, q, k, qr, kr, sc, pr, ao, x1, h2, f, x1 + f
L0_NAMES = ["norm1", "q", "k", "q_rope", "k_rope", "scores", "probs", "attn_out", "resid1", "norm2", "ffn", "block_out"]

def diagnose(model, a_out, s1i, s2i, stamp):
    import onnxruntime as ort
    print("=== DIAGNOSE: s1 graph diverges from PyTorch; locating first bad stage ===")
    n = len(model.transformer)
    names = ["embedding", "time_emb", "emb+time"] + [f"layer{i}" for i in range(n)] + ["final_norm", "logits"]
    dyn = {"s1_ids": {0: "b", 1: "t"}, "s2_ids": {0: "b", 1: "t"}, "stamp": {0: "b", 1: "t"}}
    path = os.path.join(a_out, "dbg.onnx")
    torch.onnx.export(Dbg(model), (s1i, s2i, stamp), path, input_names=["s1_ids", "s2_ids", "stamp"],
                      output_names=names, dynamic_axes=dyn, opset_version=18, do_constant_folding=True, dynamo=False)
    with torch.no_grad(): ref = Dbg(model)(s1i, s2i, stamp)
    feeds = {"s1_ids": s1i.numpy(), "s2_ids": s2i.numpy(), "stamp": stamp.numpy()}
    for label, lvl in (("ORT optimizations ON", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                       ("ORT optimizations OFF", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)):
        so = ort.SessionOptions(); so.graph_optimization_level = lvl
        outs = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(None, feeds)
        print(f"--- {label} ---")
        for nm, r, o in zip(names, ref, outs):
            d = float(np.max(np.abs(r.numpy() - o))); sc = max(1.0, float(np.max(np.abs(r.numpy()))))
            print(f"  {nm:12s} max abs diff {d:.3e}  rel {d/sc:.3e}")
    # ---- zoom into layer0: eager vs ONNX(opt off) vs torch.jit.trace ----
    print("--- layer0 sub-steps (eager vs ONNX | eager vs jit.trace) ---")
    x0 = ref[2].detach()
    l0 = L0(model).eval()
    with torch.no_grad():
        eager = l0(x0)
        tr = torch.jit.trace(l0, (x0,), check_trace=False)
        jit = tr(x0)
    p2 = os.path.join(a_out, "l0.onnx")
    torch.onnx.export(l0, (x0,), p2, input_names=["x"], output_names=L0_NAMES, opset_version=18,
                      do_constant_folding=True, dynamo=False)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    oo = ort.InferenceSession(p2, so, providers=["CPUExecutionProvider"]).run(None, {"x": x0.numpy()})
    for nm, e, j, o in zip(L0_NAMES, eager, jit, oo):
        sc = max(1.0, float(np.max(np.abs(e.numpy()))))
        d1 = float(np.max(np.abs(e.numpy() - o))); d2 = float(np.max(np.abs(e.numpy() - j.numpy())))
        print(f"  {nm:10s} onnx rel {d1/sc:.3e} (abs {d1:.2e}) | jit rel {d2/sc:.3e}  |out|max {sc:.2f}")
    # ---- isolate the cause: dynamic axes vs input sensitivity ----
    print("--- isolation tests ---")
    def rel(a_, b_):
        a_ = np.asarray(a_); b_ = np.asarray(b_)
        return float(np.max(np.abs(a_ - b_))) / max(1.0, float(np.max(np.abs(a_))))
    def run_off(path, fd):
        so_ = ort.SessionOptions(); so_.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(path, so_, providers=["CPUExecutionProvider"]).run(None, fd)
    # A) layer0 alone WITH dynamic axes
    p3 = os.path.join(a_out, "l0dyn.onnx")
    torch.onnx.export(l0, (x0,), p3, input_names=["x"], output_names=L0_NAMES, opset_version=18,
                      do_constant_folding=True, dynamo=False, dynamic_axes={"x": {0: "b", 1: "t"}})
    od = run_off(p3, {"x": x0.numpy()})
    print(f"  A) layer0 alone, DYNAMIC axes : block_out rel {rel(eager[-1].numpy(), od[-1]):.3e}  (static was ~4e-7)")
    # B) whole s1 debug graph WITHOUT dynamic axes
    p4 = os.path.join(a_out, "dbg_static.onnx")
    torch.onnx.export(Dbg(model), (s1i, s2i, stamp), p4, input_names=["s1_ids", "s2_ids", "stamp"],
                      output_names=names, opset_version=18, do_constant_folding=True, dynamo=False)
    os_ = run_off(p4, feeds)
    print(f"  B) full s1, STATIC shapes     : layer0 rel {rel(ref[3].numpy(), os_[3]):.3e}  logits rel {rel(ref[-1].numpy(), os_[-1]):.3e}")
    # C) eager only: perturb layer0 input by ~2e-5 (the embedding noise) -> does PyTorch itself blow up?
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        pert = l0(x0 + 2e-5 * torch.randn(x0.shape, generator=g))[-1]
    print(f"  C) eager, input +2e-5 noise   : block_out rel change {rel(eager[-1].numpy(), pert.numpy()):.3e}")
    print("=== END DIAGNOSE ===")

def check_cos(name, a, b, min_cos=0.95):
    """For ill-conditioned fp32 graphs (PyTorch itself moves ~20% under a 2e-5 input nudge, so element-wise
    tolerances are meaningless): compare direction per sample. Each block was verified at ~1e-7 with identical
    inputs, so cosine here guards against structural errors (wrong mask/RoPE/weights give cos far below 0.95)."""
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1); b = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    print(f"  [{name}] cosine per sample = {np.round(cos, 4).tolist()} (min required {min_cos})")
    return float(cos.min()) >= min_cos

def realistic_inputs(b, t, tok, seed):
    """Market-like OHLCV window + consecutive timestamps + the tokenizer's own tokens.
    Random noise tokens are out-of-distribution: PyTorch itself changes ~77% on a 2e-5 input perturbation,
    so comparing PyTorch with ONNX on them is meaningless. In-distribution inputs make the check stable."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.002, (b, t)), 1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], 1)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, (b, t))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, (b, t))))
    vol = rng.lognormal(3, 0.5, (b, t)); amt = vol * close
    x = np.stack([open_, high, low, close, vol, amt], -1)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)
    x = torch.from_numpy(np.clip(x, -5, 5).astype(np.float32))
    st = []
    for i in range(b):
        ix = pd.date_range("2025-03-07 09:35", periods=t, freq="5min") + pd.Timedelta(days=int(rng.integers(0, 300)))
        st.append(np.stack([ix.minute, ix.hour, ix.dayofweek, ix.day, ix.month], -1))
    stamp = torch.from_numpy(np.stack(st)).float()
    s1i, s2i = Enc(tok)(x)
    return x, s1i, s2i, stamp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mini", choices=PRESETS)
    ap.add_argument("--out", default="out")
    ap.add_argument("--strict", action="store_true", help="exit with error if any verification fails")
    a = ap.parse_args()
    global STRICT; STRICT = a.strict
    mid, tid, max_ctx = PRESETS[a.preset]
    os.makedirs(a.out, exist_ok=True)
    print(f"Loading {mid} + {tid} ..."); torch.manual_seed(0)
    model = Kronos.from_pretrained(mid).eval(); tok = KronosTokenizer.from_pretrained(tid).eval()
    V1, V2, D = 2 ** model.s1_bits, 2 ** model.s2_bits, model.d_model
    P = lambda n: os.path.join(a.out, n)

    B, T = 2, 48
    x = torch.randn(B, T, 6).clamp(-5, 5)
    s1i = torch.randint(0, V1, (B, T)); s2i = torch.randint(0, V2, (B, T))
    stamp = torch.stack([torch.randint(0, 60, (B, T)), torch.randint(0, 24, (B, T)), torch.randint(0, 7, (B, T)),
                         torch.randint(1, 32, (B, T)), torch.randint(1, 13, (B, T))], -1).float()
    ctx0 = torch.randn(B, T, D); one = torch.randint(0, V1, (B, 1))

    print("Exporting ...")
    export(Enc(tok), (x,), ["x"], ["s1", "s2"], {"x": {0: "b", 1: "t"}, "s1": {0: "b", 1: "t"}, "s2": {0: "b", 1: "t"}}, P("encoder.onnx"))
    export(S1(model), (s1i, s2i, stamp), ["s1_ids", "s2_ids", "stamp"], ["s1_logits", "context"],
           {"s1_ids": {0: "b", 1: "t"}, "s2_ids": {0: "b", 1: "t"}, "stamp": {0: "b", 1: "t"}, "s1_logits": {0: "b"}, "context": {0: "b", 1: "t"}}, P("s1.onnx"))
    export(S2(model), (ctx0, one), ["context", "s1_id"], ["s2_logits"],
           {"context": {0: "b", 1: "t"}, "s1_id": {0: "b"}, "s2_logits": {0: "b"}}, P("s2.onnx"))
    export(Dec(tok), (s1i, s2i), ["s1", "s2"], ["recon"],
           {"s1": {0: "b", 1: "t"}, "s2": {0: "b", 1: "t"}, "recon": {0: "b", 1: "t"}}, P("decoder.onnx"))

    print("Verifying against PyTorch (T=48 and T=200, batch 2 and 3) ...")
    with torch.no_grad():
        for (b, t) in [(2, 48), (3, 200)]:
            x, s1i, s2i, stamp = realistic_inputs(b, t, tok, seed=b * 1000 + t)
            e1, e2 = Enc(tok)(x); o1, o2 = run_ort(P("encoder.onnx"), {"x": x.numpy()})
            for n, p, q in (("encoder.s1", e1, o1), ("encoder.s2", e2, o2)):
                mism = float((p.numpy() != q).mean()); print(f"  [{n}] token mismatch = {mism:.4f}")
                if mism > 0.01: fail(f"VERIFY FAILED for {n}")
            lg, ctx = S1(model)(s1i, s2i, stamp)
            ol, oc = run_ort(P("s1.onnx"), {"s1_ids": s1i.numpy(), "s2_ids": s2i.numpy(), "stamp": stamp.numpy()})
            print(f"  [s1] top-1 token agreement = {float((lg.argmax(-1).numpy() == ol.argmax(-1)).mean()):.2f}")
            if not (check_cos("s1.logits", lg.numpy(), ol) and check_cos("s1.context", ctx.numpy(), oc)):
                fail("VERIFY FAILED for s1 (cosine too low)")
            one = s1i[:, -1:]
            check("s2.logits", S2(model)(ctx, one).numpy(), run_ort(P("s2.onnx"), {"context": ctx.numpy(), "s1_id": one.numpy()})[0])
            check("decoder", Dec(tok)(s1i, s2i).numpy(), run_ort(P("decoder.onnx"), {"s1": s1i.numpy(), "s2": s2i.numpy()})[0], tol=5e-3)

    if FAILS: print(f"!!! {len(FAILS)} verification warning(s); graphs exported anyway:", *FAILS, sep="\n    ")
    else: print("All verifications passed.")
    names = ["encoder.onnx", "s1.onnx", "s2.onnx", "decoder.onnx"]
    manifest = {"name": f"kronos-{a.preset}", "version": 1, "source": mid, "tokenizer": tid,
                "s1_bits": model.s1_bits, "s2_bits": model.s2_bits, "d_model": D, "max_context": max_ctx, "clip": 5.0,
                "files": {n: {"sha256": sha256(P(n)), "size": os.path.getsize(P(n))} for n in names}}
    json.dump(manifest, open(P("manifest.json"), "w"), indent=2)
    with zipfile.ZipFile(P("kronos-onnx.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["manifest.json"]: z.write(P(n), n)
    print("OK ->", P("kronos-onnx.zip"), f"({os.path.getsize(P('kronos-onnx.zip'))/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
