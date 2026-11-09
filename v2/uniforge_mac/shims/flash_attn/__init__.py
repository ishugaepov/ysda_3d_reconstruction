"""Stand-in for flash-attn on Apple Silicon: same call signatures, exact attention via torch SDPA.

Queries are processed in chunks so memory stays O(chunk x keys) even for tens of thousands of
sparse tokens (MPS SDPA may materialise the full score matrix).
"""
import os

import torch
import torch.nn.functional as F

_CHUNK = int(os.environ.get("YSDA_ATTN_Q_CHUNK", "2048"))


def _sdpa_bnhc(q, k, v, softmax_scale=None):
    """q [B,Nq,H,C], k/v [B,Nk,H,C] -> [B,Nq,H,C]."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # [B,H,N,C]
    outs = []
    for a in range(0, q.shape[2], _CHUNK):
        outs.append(F.scaled_dot_product_attention(q[:, :, a:a + _CHUNK], k, v, scale=softmax_scale))
    return torch.cat(outs, 2).transpose(1, 2).contiguous()


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    assert not causal, "causal attention not needed by UniForge3D"
    return _sdpa_bnhc(q, k, v, softmax_scale)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    """q [Tq,H,C], k/v [Tk,H,C], cu_seqlens = segment boundaries -> [Tq,H,C]."""
    assert not causal
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for s in range(len(cq) - 1):
        qa, qb, ka, kb = cq[s], cq[s + 1], ck[s], ck[s + 1]
        if qb > qa:
            out[qa:qb] = _sdpa_bnhc(q[None, qa:qb], k[None, ka:kb], v[None, ka:kb], softmax_scale)[0]
    return out


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None,
                                     causal=False, **kw):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                  softmax_scale=softmax_scale, causal=causal)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                    dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                  softmax_scale=softmax_scale, causal=causal)
