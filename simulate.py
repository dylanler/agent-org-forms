#!/usr/bin/env python3
"""Numpy-only organizational-mix simulation. No LLM calls. Seeded."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/workspace/org-wisdom")
PLOT_DIR = OUT / "plots"

N_AGENTS = 48
K_DOM = 8
T_MAIN = 200
SHOCK_T = 100
PROMOTE_T = 80
N_SEEDS_MAIN = 30
N_SEEDS_SWEEP = 10
SIGMA = 0.5
ALPHA_MAIN = 1.0
GAMMA_MAIN = -0.14
ETA_C = 2.0
ETA_K = 2.0
DELTA_IDLE = 1.0
IOTA0 = 0.5
BETA0 = 0.0
BETA2 = 1.0
BETA1_MAIN = -1.0
CROSS_LAMBDA = 0.05
N_BLOCKS = 6
BLOCK_SIZE = 8
CROSS_PAIRS = N_BLOCKS * (N_BLOCKS - 1) / 2.0
CROSS_LOAD = CROSS_LAMBDA * CROSS_PAIRS

PI_C_GRID = [0.0, 0.0625, 0.125, 0.25, 0.5]
SPEC_GRID = [0.25, 0.5, 0.75]
PI_K_GRID = [0.0, 0.1]
STRUCTURES = ["modules", "flat", "modules_isolated"]

MU = np.array(
    [
        [2.0, -1.0, -1.0, 0.0, 0.0],
        [0.0, 2.0, -0.5, 0.5, 0.0],
        [-0.5, 0.0, 2.0, 0.5, 0.0],
        [-1.0, 0.5, 0.0, 2.0, 0.0],
    ],
    dtype=float,
)
ROLE_NAME = ["S", "G", "C", "K"]


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalized_entropy(p):
    pc = np.clip(p, 1e-12, None)
    pc = pc / pc.sum(axis=-1, keepdims=True)
    h = -np.sum(pc * np.log(pc), axis=-1) / np.log(p.shape[-1])
    return np.clip(h, 0.0, 1.0)


def make_pi(pi_C, specialist_share, pi_K):
    if pi_C + pi_K > 1.0 + 1e-12:
        raise ValueError("pi_C + pi_K > 1")
    maker = 1.0 - pi_C - pi_K
    pi = np.array(
        [
            specialist_share * maker,
            (1.0 - specialist_share) * maker,
            pi_C,
            pi_K,
        ],
        dtype=float,
    )
    return pi / pi.sum()


def largest_remainder(pi, n):
    raw = np.round(np.asarray(pi, dtype=float) * n, decimals=10)
    base = np.floor(raw).astype(int)
    rem = int(n - int(base.sum()))
    if rem < 0:
        raise RuntimeError(f"negative remainder {rem} for {pi}")
    frac = raw - base
    order = np.lexsort((np.arange(len(pi)), -frac))
    if rem > 0:
        base[order[:rem]] += 1
    if int(base.sum()) != n:
        raise RuntimeError(f"counts {base} do not sum to {n}")
    return base


def build_tasks(seeds, T=T_MAIN, shock_t=SHOCK_T, K=K_DOM):
    """lambda_t = 0.2 if t even else 0.8. Sparse is Bernoulli(0.5) per seed-period.
    For t >= shock_t, a seed-specific permutation moves task mass across domains.
    """
    n = len(seeds)
    q = np.zeros((T, n, K), dtype=float)
    sparse = np.zeros((T, n), dtype=bool)
    perms = np.zeros((n, K), dtype=int)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4]))
        perms[i] = rng.permutation(K)
        for t in range(T):
            is_sparse = bool(rng.random() < 0.5)
            sparse[t, i] = is_sparse
            if is_sparse:
                choice = rng.choice(K, size=2, replace=False)
                w = rng.random(2)
                w = w / w.sum()
                qt = np.zeros(K, dtype=float)
                qt[choice] = w
            else:
                g = rng.gamma(1.0, 1.0, size=K)
                qt = g / g.sum()
            if t >= shock_t:
                moved = np.zeros(K, dtype=float)
                moved[perms[i]] = qt
                qt = moved
            q[t, i] = qt
    lam = np.where(np.arange(T) % 2 == 0, 0.2, 0.8).astype(float)
    z = (np.arange(T) >= shock_t).astype(float)
    return {"q": q, "lam": lam, "z": z, "sparse": sparse, "perms": perms, "shock_t": shock_t}


def build_pop(seeds, pi, sigma=SIGMA, n=N_AGENTS, K=K_DOM):
    """Hamilton counts, then a random assignment of that multiset.
    Specialists: Dirichlet peak 20 and 0.4 elsewhere. Others: Dirichlet(1).
    """
    counts = largest_remainder(pi, n)
    n_seeds = len(seeds)
    roles = np.zeros((n_seeds, n), dtype=int)
    theta = np.zeros((n_seeds, n, 5), dtype=float)
    spec = np.zeros((n_seeds, n, K), dtype=float)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), 11, int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])]
            )
        )
        base = np.concatenate(
            [
                np.full(int(counts[0]), 0, dtype=int),
                np.full(int(counts[1]), 1, dtype=int),
                np.full(int(counts[2]), 2, dtype=int),
                np.full(int(counts[3]), 3, dtype=int),
            ]
        )
        roles[i] = rng.permutation(base)
        eps = rng.normal(0.0, sigma, size=(n, 5))
        theta[i] = sigmoid(MU[roles[i]] + eps)
        alpha = np.ones((n, K), dtype=float)
        is_S = roles[i] == 0
        if np.any(is_S):
            alpha[is_S] = 0.4
            idxs = np.flatnonzero(is_S)
            peaks = rng.integers(0, K, size=idxs.size)
            alpha[idxs, peaks] = 20.0
        g = rng.gamma(alpha, 1.0)
        spec[i] = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-15)
    H = normalized_entropy(spec)
    return {
        "role": roles,
        "theta": theta,
        "spec": spec,
        "H": H,
        "counts": counts.astype(int),
        "pi": np.asarray(pi, dtype=float),
    }


def demand_L(n_per, lam, link_mode):
    """Main loop uses Brooks. Graicunas is sensitivity only."""
    if link_mode == "brooks":
        return float(lam) * n_per * (n_per - 1) / 2.0
    if link_mode == "graicunas":
        load = n_per * (2.0 ** (n_per - 1) + n_per - 1.0)
        return float(lam) * load
    raise ValueError(link_mode)


def eval_org(m, role, theta, n_per, lam, alpha, gamma, cross_load, link_mode):
    """Per-block Brooks loss and loafing. Y is the sum across blocks times f_cross.

    L = lambda * n * (n-1) / 2
    g = max(0, L - S_C), S_C = sum of coordinator c
    factor = exp(-alpha * g / (L+1)); extra supply past L does not raise it
    bar_c = S_C / n
    bar_k = mean communicator k, else 0
    rho = sigmoid(2*bar_c + 2*bar_k)
    e = rho + (1-rho) * n**gamma
    Y_block = P * e * factor

    Maker idle: e*m strictly below half the median of e*m among makers.
    The 10th percentile is not the threshold.
    Coordinator idle mass = (S_C-L)/S_C * n_C when S_C > L.
    On modules, surplus skill used to cover L_cross is not counted idle.
    iota = (idle_coord_mass + n_idle_makers) / N
    Reported coord_loss = 1 - Y / sum_b(P_b * e_b)
    """
    n_seeds, n_agents = m.shape
    if n_agents % n_per != 0:
        raise RuntimeError("n_agents not divisible by block size")
    n_blocks = n_agents // n_per
    m_b = m.reshape(n_seeds, n_blocks, n_per)
    role_b = role.reshape(n_seeds, n_blocks, n_per)
    c_b = theta[:, :, 2].reshape(n_seeds, n_blocks, n_per)
    k_b = theta[:, :, 3].reshape(n_seeds, n_blocks, n_per)
    is_C = role_b == 2
    is_K = role_b == 3
    S_C = (c_b * is_C).sum(axis=-1)
    n_C = is_C.sum(axis=-1).astype(float)
    P_b = m_b.sum(axis=-1)
    L = demand_L(n_per, lam, link_mode)
    g = np.maximum(0.0, L - S_C)
    if L <= 0.0:
        f = np.ones_like(g)
    else:
        f = np.exp(-alpha * g / (L + 1.0))
    bar_c = S_C / float(n_per)
    n_K = is_K.sum(axis=-1).astype(float)
    k_sum = (k_b * is_K).sum(axis=-1)
    bar_k = np.divide(k_sum, n_K, out=np.zeros_like(k_sum), where=n_K > 0)
    rho = sigmoid(ETA_C * bar_c + ETA_K * bar_k)
    n_pow = float(n_per) ** float(gamma)
    e = rho + (1.0 - rho) * n_pow
    Y_pre = (P_b * e).sum(axis=-1)
    Y_blocks = (P_b * e * f).sum(axis=-1)
    surplus = np.maximum(0.0, S_C - L)
    if cross_load > 0.0:
        S_cross = surplus.sum(axis=-1)
        g_cross = np.maximum(0.0, cross_load - S_cross)
        f_cross = np.exp(-alpha * g_cross / (cross_load + 1.0))
    else:
        f_cross = 1.0
    Y = Y_blocks * f_cross
    idle_block = np.where(
        (S_C > L) & (n_C > 0.0),
        (S_C - L) / np.maximum(S_C, 1e-15) * n_C,
        0.0,
    )
    if cross_load > 0.0:
        total_surplus = surplus.sum(axis=-1)
        used = np.minimum(float(cross_load), total_surplus)
        frac_used = np.divide(
            used,
            total_surplus,
            out=np.zeros_like(total_surplus),
            where=total_surplus > 1e-12,
        )
        idle_c = idle_block.sum(axis=-1) * (1.0 - frac_used)
    else:
        idle_c = idle_block.sum(axis=-1)
    e_agent = np.repeat(e, n_per, axis=1)
    em = e_agent * m
    is_maker = (role == 0) | (role == 1)
    em_masked = np.where(is_maker, em, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(em_masked, axis=1)
    half = 0.5 * med
    idle_m = np.zeros(n_seeds, dtype=float)
    ok = np.isfinite(half)
    if np.any(ok):
        idle_m[ok] = np.sum((em[ok] < half[ok, None]) & is_maker[ok], axis=1).astype(float)
    iota = (idle_c + idle_m) / float(n_agents)
    Y_net = Y * (1.0 - DELTA_IDLE * np.maximum(0.0, iota - IOTA0))
    eff = np.divide(Y, Y_pre, out=np.ones_like(Y), where=Y_pre > 1e-15)
    coord_loss = 1.0 - eff
    mean_e = e_agent.mean(axis=1)
    return Y, iota, coord_loss, mean_e, Y_net


def agreement(belief, code, is_maker):
    b_arg = np.argmax(belief, axis=-1)
    c_arg = np.argmax(code, axis=-1)
    match = (b_arg == c_arg[:, None]) & is_maker
    n_m = is_maker.sum(axis=1).astype(float)
    u = np.zeros(belief.shape[0], dtype=float)
    ok = n_m > 0
    u[ok] = match[ok].sum(axis=1) / n_m[ok]
    return u


def update_code(belief, code, spec, m, role, theta, q, sdot):
    """Not March's exact recursion.

    p2 = 0.05 + 0.2 * mean(a of makers with s·q > code·q); empty => 0
    vote = sum(m_i s_i) / sum(m_i) over makers
    code <- renormalize((1-p2)*code + p2*vote)
    p1 = 0.02 + 0.5 * mean(k of communicators); none => 0.02
    belief <- (1-p1)*belief + p1*code for every maker
    u is measured before this update. Production uses s, not belief.
    """
    is_maker = (role == 0) | (role == 1)
    is_K = role == 3
    code_dot = np.einsum("nk,nk->n", code, q)
    a = theta[:, :, 4]
    better = (sdot > code_dot[:, None]) & is_maker
    n_better = better.sum(axis=1).astype(float)
    sum_a = (a * better).sum(axis=1)
    mean_a = np.zeros(code.shape[0], dtype=float)
    okb = n_better > 0
    mean_a[okb] = sum_a[okb] / n_better[okb]
    p2 = 0.05 + 0.2 * mean_a
    m_maker = np.where(is_maker, m, 0.0)
    m_sum = m_maker.sum(axis=1)
    vote = np.full_like(code, 1.0 / code.shape[1])
    okm = m_sum > 1e-15
    if np.any(okm):
        vote[okm] = np.einsum("ni,nik->nk", m_maker[okm], spec[okm]) / m_sum[okm, None]
    has = is_maker.any(axis=1)
    code_new = (1.0 - p2)[:, None] * code + p2[:, None] * vote
    code_new = np.clip(code_new, 0.0, None)
    zsum = code_new.sum(axis=1, keepdims=True)
    code_new = np.divide(code_new, np.maximum(zsum, 1e-15))
    code = np.where(has[:, None], code_new, code)
    k = theta[:, :, 3]
    n_K = is_K.sum(axis=1).astype(float)
    k_sum = np.where(is_K, k, 0.0).sum(axis=1)
    mean_k = np.zeros(code.shape[0], dtype=float)
    okk = n_K > 0
    mean_k[okk] = k_sum[okk] / n_K[okk]
    p1 = np.clip(0.02 + 0.5 * mean_k, 0.0, 1.0)
    belief = np.where(
        is_maker[:, :, None],
        (1.0 - p1)[:, None, None] * belief + p1[:, None, None] * code[:, None, :],
        belief,
    )
    return belief, code


def apply_promotion(kind, role, theta, m_recent, beta1):
    """Peter and Match promote 4 makers to C and set d=0.
    c = sigmoid(beta0 + beta1*d + beta2*b), beta0=0, beta2=1.
    Tournament: no role change; top quartile of makers by recent m get +0.15 logit(a).
    """
    n_seeds = role.shape[0]
    is_maker = (role == 0) | (role == 1)
    s_idx = np.arange(n_seeds)[:, None]
    if kind in ("peter", "match"):
        trait = theta[:, :, 0] if kind == "peter" else theta[:, :, 1]
        score = np.where(is_maker, trait, -np.inf)
        n_m = is_maker.sum(axis=1)
        if np.any(n_m < 1):
            raise RuntimeError("no makers to promote")
        k = 4 if np.all(n_m >= 4) else int(n_m.min())
        idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        if not np.all(is_maker[s_idx, idx]):
            raise RuntimeError("promotion selected a non-maker")
        d_old = theta[s_idx, idx, 0]
        b_old = theta[s_idx, idx, 1]
        c_new = sigmoid(BETA0 + beta1 * d_old + BETA2 * b_old)
        theta[s_idx, idx, 2] = c_new
        theta[s_idx, idx, 0] = 0.0
        role[s_idx, idx] = 2
        return k
    if kind == "tournament":
        score = np.where(is_maker, m_recent, -np.inf)
        n_m = is_maker.sum(axis=1)
        if np.any(n_m != n_m[0]):
            raise RuntimeError("maker counts differ across seeds")
        k = max(1, int(np.floor(float(n_m[0]) * 0.25)))
        idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        if not np.all(is_maker[s_idx, idx]):
            raise RuntimeError("tournament selected a non-maker")
        a = np.clip(theta[s_idx, idx, 4], 1e-6, 1.0 - 1e-6)
        logit = np.log(a) - np.log(1.0 - a)
        theta[s_idx, idx, 4] = sigmoid(logit + 0.15)
        return k
    raise ValueError(kind)


def _role_mean(values, role, r):
    mask = role == r
    cnt = mask.sum(axis=1).astype(float)
    sm = (values * mask).sum(axis=1)
    out = np.full(role.shape[0], np.nan, dtype=float)
    ok = cnt > 0
    out[ok] = sm[ok] / cnt[ok]
    return out


def _safe_div(sm, cnt):
    out = np.full_like(sm, np.nan, dtype=float)
    ok = cnt > 0
    out[ok] = sm[ok] / cnt[ok]
    return out


def run_from_pop(
    pop,
    tasks,
    seed_index,
    alpha,
    gamma,
    structures,
    promotion=None,
    beta1=BETA1_MAIN,
    promote_t=PROMOTE_T,
    link_mode="brooks",
):
    """z_weight is 0 before the shock (breadth only) and 1 after (depth only)."""
    sl = np.asarray(seed_index, dtype=int)
    role = pop["role"][sl].copy()
    theta = pop["theta"][sl].copy()
    spec = pop["spec"][sl]
    H = pop["H"][sl]
    q_all = tasks["q"][:, sl]
    lam = tasks["lam"]
    z = tasks["z"]
    T, n_seeds, _K = q_all.shape
    one_m_H = 1.0 - H
    sdot_all = np.einsum("tnk,nik->tni", q_all, spec)

    init = {}
    for r, name in enumerate(ROLE_NAME):
        init[f"init_H_{name}"] = _role_mean(H, role, r)
        init[f"init_d_{name}"] = _role_mean(theta[:, :, 0], role, r)
        init[f"init_b_{name}"] = _role_mean(theta[:, :, 1], role, r)
        init[f"init_breadth_{name}"] = _role_mean(theta[:, :, 1] * one_m_H, role, r)
        init[f"init_k_{name}"] = _role_mean(theta[:, :, 3], role, r)

    code = np.full((n_seeds, K_DOM), 1.0 / K_DOM, dtype=float)
    belief = spec.copy()
    buf = np.zeros((10, n_seeds, N_AGENTS), dtype=float)
    buf_i = 0
    filled = 0

    struct_specs = []
    for name in structures:
        if name == "flat":
            struct_specs.append({"name": name, "n_per": N_AGENTS, "cross": 0.0})
        elif name == "modules":
            struct_specs.append({"name": name, "n_per": BLOCK_SIZE, "cross": float(CROSS_LOAD)})
        elif name == "modules_isolated":
            struct_specs.append({"name": name, "n_per": BLOCK_SIZE, "cross": 0.0})
        else:
            raise ValueError(name)

    keys_sum = [
        "cum_D",
        "cum_R",
        "cum_D_highlam",
        "cum_R_highlam",
        "cum_D_lowlam",
        "cum_R_lowlam",
        "sum_iota",
        "sum_coord_loss",
        "sum_e",
        "sum_Y",
        "sum_Y_net",
        "sum_u_pre",
        "sum_u_post",
        "sum_Y_from80",
        "sum_coord_from80",
        "sum_e_from80",
        "sum_D_from80",
        "sum_R_from80",
        "n_from80",
    ]
    acc = {sp["name"]: {k: np.zeros(n_seeds, dtype=float) for k in keys_sum} for sp in struct_specs}
    iota_hist = {sp["name"]: np.zeros((T, n_seeds), dtype=float) for sp in struct_specs}
    m_sum_pre = np.zeros((n_seeds, 4), dtype=float)
    m_cnt_pre = np.zeros((n_seeds, 4), dtype=float)
    m_sum_post = np.zeros((n_seeds, 4), dtype=float)
    m_cnt_post = np.zeros((n_seeds, 4), dtype=float)
    promoted_k = 0

    for t in range(T):
        if promotion is not None and t == promote_t:
            if filled < 10:
                raise RuntimeError("promotion needs 10 prior periods")
            recent = buf.mean(axis=0)
            promoted_k = apply_promotion(promotion, role, theta, recent, beta1)
        d = theta[:, :, 0]
        b = theta[:, :, 1]
        a = theta[:, :, 4]
        z_t = float(z[t])
        if z_t == 0.0:
            m = a * (b * one_m_H)
        else:
            m = a * (d * sdot_all[t])
        is_maker = (role == 0) | (role == 1)
        u = agreement(belief, code, is_maker)
        if not np.isfinite(m).all() or not np.isfinite(u).all():
            raise RuntimeError(f"non-finite m or u at t={t}")
        high = float(lam[t]) >= 0.5
        for r in range(4):
            mask = role == r
            if z_t == 0.0:
                m_sum_pre[:, r] += (m * mask).sum(axis=1)
                m_cnt_pre[:, r] += mask.sum(axis=1)
            else:
                m_sum_post[:, r] += (m * mask).sum(axis=1)
                m_cnt_post[:, r] += mask.sum(axis=1)
        for sp in struct_specs:
            Y, iota, coord_loss, mean_e, Y_net = eval_org(
                m, role, theta, sp["n_per"], float(lam[t]), alpha, gamma, sp["cross"], link_mode
            )
            if z_t == 0.0:
                D = np.zeros_like(Y)
                R = Y * u
            else:
                D = Y * (1.0 - u)
                R = np.zeros_like(Y)
            if not np.isfinite(Y).all():
                raise RuntimeError(f"non-finite Y at t={t} structure={sp['name']}")
            bucket = acc[sp["name"]]
            bucket["cum_D"] += D
            bucket["cum_R"] += R
            if high:
                bucket["cum_D_highlam"] += D
                bucket["cum_R_highlam"] += R
            else:
                bucket["cum_D_lowlam"] += D
                bucket["cum_R_lowlam"] += R
            bucket["sum_iota"] += iota
            bucket["sum_coord_loss"] += coord_loss
            bucket["sum_e"] += mean_e
            bucket["sum_Y"] += Y
            bucket["sum_Y_net"] += Y_net
            if z_t == 0.0:
                bucket["sum_u_pre"] += u
            else:
                bucket["sum_u_post"] += u
            if t >= promote_t:
                bucket["sum_Y_from80"] += Y
                bucket["sum_coord_from80"] += coord_loss
                bucket["sum_e_from80"] += mean_e
                bucket["sum_D_from80"] += D
                bucket["sum_R_from80"] += R
                bucket["n_from80"] += 1.0
            iota_hist[sp["name"]][t] = iota
        belief, code = update_code(belief, code, spec, m, role, theta, q_all[t], sdot_all[t])
        buf[buf_i] = m
        buf_i = (buf_i + 1) % 10
        filled += 1

    n_pre = int(np.sum(z < 0.5))
    n_post = int(np.sum(z >= 0.5))
    if n_pre <= 0 or n_post <= 0:
        raise RuntimeError("shock split is empty")
    out = {}
    for sp in struct_specs:
        bucket = acc[sp["name"]]
        n_from = float(bucket["n_from80"][0])
        if n_from <= 0:
            raise RuntimeError("post-promotion window empty")
        metrics = {
            "cum_D": bucket["cum_D"],
            "cum_R": bucket["cum_R"],
            "cum_D_highlam": bucket["cum_D_highlam"],
            "cum_R_highlam": bucket["cum_R_highlam"],
            "cum_D_lowlam": bucket["cum_D_lowlam"],
            "cum_R_lowlam": bucket["cum_R_lowlam"],
            "mean_iota": bucket["sum_iota"] / float(T),
            "median_iota": np.median(iota_hist[sp["name"]], axis=0),
            "mean_coord_loss": bucket["sum_coord_loss"] / float(T),
            "mean_e": bucket["sum_e"] / float(T),
            "mean_Y": bucket["sum_Y"] / float(T),
            "mean_Y_net": bucket["sum_Y_net"] / float(T),
            "mean_u_pre": bucket["sum_u_pre"] / float(n_pre),
            "mean_u_post": bucket["sum_u_post"] / float(n_post),
            "post_mean_Y": bucket["sum_Y_from80"] / n_from,
            "post_mean_coord_loss": bucket["sum_coord_from80"] / n_from,
            "post_mean_e": bucket["sum_e_from80"] / n_from,
            "post_D": bucket["sum_D_from80"],
            "post_R": bucket["sum_R_from80"],
        }
        if not np.allclose(bucket["cum_D"], bucket["cum_D_highlam"] + bucket["cum_D_lowlam"]):
            raise RuntimeError("D split mismatch")
        if not np.allclose(bucket["cum_R"], bucket["cum_R_highlam"] + bucket["cum_R_lowlam"]):
            raise RuntimeError("R split mismatch")
        for r, name in enumerate(ROLE_NAME):
            metrics[f"mean_m_pre_{name}"] = _safe_div(m_sum_pre[:, r], m_cnt_pre[:, r])
            metrics[f"mean_m_post_{name}"] = _safe_div(m_sum_post[:, r], m_cnt_post[:, r])
            for ikey, arr in init.items():
                if ikey.endswith("_" + name):
                    metrics[ikey] = arr
        for key, arr in metrics.items():
            if not np.isfinite(arr).all():
                if key.startswith("mean_m_") or key.startswith("init_"):
                    continue
                raise RuntimeError(f"non-finite {key} in {sp['name']}")
        metrics["_promoted_k"] = promoted_k
        out[sp["name"]] = metrics
    return out


def pack_strict(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("pack_strict got non-finite data")
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {"mean": float(x.mean()), "std": std}


def pack_maybe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "std": None}
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {"mean": float(x.mean()), "std": std}


CORE_KEYS = [
    "cum_D",
    "cum_R",
    "cum_D_highlam",
    "cum_R_highlam",
    "cum_D_lowlam",
    "cum_R_lowlam",
    "mean_iota",
    "median_iota",
    "mean_coord_loss",
    "mean_e",
    "mean_Y",
    "mean_Y_net",
    "mean_u_pre",
    "mean_u_post",
    "post_mean_Y",
    "post_mean_coord_loss",
    "post_mean_e",
    "post_D",
    "post_R",
]


def summarize_run(metrics):
    rec = {}
    for key in CORE_KEYS:
        rec[key] = pack_strict(metrics[key])
    med = np.asarray(metrics["median_iota"], dtype=float)
    rec["frac_seeds_median_iota_ge_0.8"] = float(np.mean(med >= 0.8))
    rec["median_across_seeds_of_median_iota"] = float(np.median(med))
    rec["max_seed_median_iota"] = float(np.max(med))
    for rname in ROLE_NAME:
        for prefix in (
            "mean_m_pre_",
            "mean_m_post_",
            "init_H_",
            "init_d_",
            "init_b_",
            "init_breadth_",
            "init_k_",
        ):
            rec[prefix + rname] = pack_maybe(metrics[prefix + rname])
    return rec


def cfg_key(pi_C, spec, pi_K):
    return f"pc={pi_C}|ss={spec}|pk={pi_K}"


def base_configs():
    configs = []
    for pi_C in PI_C_GRID:
        for spec in SPEC_GRID:
            for pi_K in PI_K_GRID:
                configs.append(
                    {
                        "pi_C": pi_C,
                        "specialist_share": spec,
                        "pi_K": pi_K,
                        "pi": make_pi(pi_C, spec, pi_K),
                    }
                )
    return configs


def attach_identity(rec, cfg, structure, n_seeds, counts, extra=None):
    rec = dict(rec)
    rec["structure"] = structure
    rec["pi_C"] = cfg["pi_C"]
    rec["specialist_share"] = cfg["specialist_share"]
    rec["pi_K"] = cfg["pi_K"]
    rec["pi_vector"] = [float(x) for x in cfg["pi"]]
    rec["counts"] = [int(c) for c in counts]
    rec["n_seeds"] = int(n_seeds)
    if extra:
        rec.update(extra)
    return rec


def fmt(x):
    if x is None:
        return "NA"
    x = float(x)
    if not np.isfinite(x):
        return "NA"
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax >= 0.01:
        return f"{x:.4f}"
    return f"{x:.4e}"


def pm(stat):
    if stat is None or stat.get("mean") is None:
        return "NA"
    return f"{fmt(stat['mean'])} ± {fmt(stat['std'])}"


def find_rec(records, structure, pi_C, spec, pi_K):
    for r in records:
        if (
            r["structure"] == structure
            and r["pi_C"] == pi_C
            and r["specialist_share"] == spec
            and r["pi_K"] == pi_K
        ):
            return r
    raise KeyError((structure, pi_C, spec, pi_K))


def curve_label(values):
    v = np.asarray(values, dtype=float)
    i = int(np.argmax(v))
    interior = 0 < i < len(v) - 1
    ends_below = bool(v[0] < v[i] and v[-1] < v[i])
    left = all(v[j] <= v[j + 1] + 1e-9 for j in range(i))
    right = all(v[j] >= v[j + 1] - 1e-9 for j in range(i, len(v) - 1))
    if interior and left and right:
        lab = "inverse-U (interior peak; nondecreasing up to the peak and nonincreasing after)"
    elif interior and ends_below:
        lab = "interior peak with both endpoints below the peak, but not monotone on both sides"
    elif i == 0:
        lab = "peak at the lowest pi_C (not an inverse-U)"
    else:
        lab = "peak at the highest pi_C (not an inverse-U)"
    return lab, float(PI_C_GRID[i]), i


def rank_order(records_for_alpha_gamma):
    items = sorted(records_for_alpha_gamma, key=lambda r: (-r["cum_R"]["mean"], r["pi_C"]))
    return [r["pi_C"] for r in items]


def write_readme():
    text = f"""# Organizational-mix simulation

Numpy only. No fitted parameters. Seeds are integers 0..29. Sweep and beta1 runs use seeds 0..9. Sample standard deviations use ddof=1 across seeds. Primary discovery and revenue use Y, not Y_net.

## Roles and headcount

pi = (pi_S, pi_G, pi_C, pi_K) on the simplex. N = {N_AGENTS}.
Integer counts are the largest-remainder (Hamilton) apportionment of pi*N. Ties break toward the higher fractional part, then the lower role index (S, then G, then C, then K). Agents are a random permutation of that multiset, so every seed in a cell has the same counts.

maker_pool = 1 - pi_C - pi_K
pi_S = specialist_share * maker_pool
pi_G = (1 - specialist_share) * maker_pool

## Personality and specialty

theta = (d, b, c, k, a) = sigmoid(mu_role + eps), eps ~ N(0, sigma^2 I), sigma = {SIGMA}.
sigmoid(x) = 1/(1+exp(-x)).

Logit prototypes (d, b, c, k, a):
- specialist (2, -1, -1, 0, 0)
- generalist (0, 2, -0.5, 0.5, 0)
- coordinator (-0.5, 0, 2, 0.5, 0)
- communicator (-1, 0.5, 0, 2, 0)

Specialty on K = {K_DOM} domains:
- specialists: Dirichlet with concentration 20 on one random domain and 0.4 elsewhere
- generalists, coordinators, communicators: symmetric Dirichlet(1)

H(s) = (-sum s_j log s_j) / log(K) in [0, 1].

## Tasks, shock, making

T = {T_MAIN}. Shock at t = {SHOCK_T}: z = 0 before, z = 1 at and after.
lambda_t = 0.2 on even t and 0.8 on odd t, shared across seeds.
Each seed-period is sparse with probability 0.5, independent of lambda. Sparse q uses two random domains. Dense q is Dirichlet(1).
For t >= {SHOCK_T}, a seed-specific permutation moves task mass so a specialist spike can miss.

z_weight choice: breadth term only before the shock, depth term only at and after the shock. No partial mix.

m = a * b * (1 - H(s))    if z = 0
m = a * d * (s · q)       if z = 1

Production uses true specialty s, not belief. P = sum of m over all members of the team.

## Coordination, loafing, structures

L = lambda_t * n * (n-1) / 2. This is Brooks. Graicunas is not in the main loop.
S_C = sum of c over coordinators. g = max(0, L - S_C).
factor = exp(-alpha * g / (L+1)). If L = 0 the factor is 1. Extra coordinators past L do not raise the factor.

bar_c = S_C / n. bar_k = mean k of communicators, else 0.
rho = sigmoid(2*bar_c + 2*bar_k). eta_C = eta_K = 2, unfitted.
e = rho + (1-rho) * (n ** gamma), same e for every member.
Main-grid gamma = {GAMMA_MAIN} (Latané anchor, not a universal constant). Main-grid alpha = {ALPHA_MAIN}.

Y_team = P * e * factor.

Three structures, not the same cell:
- flat: one team of 48. No cross term.
- modules: six blocks of 8. Within-block L uses lambda_t. Cross-block load L_cross = {CROSS_LAMBDA} * (6*5/2) = {CROSS_LOAD}, not scaled by lambda_t. Surplus coordinator skill after within-block demand can offset L_cross. Y = f_cross * sum of block outputs, with f_cross = exp(-alpha * g_cross / (L_cross+1)).
- modules_isolated: same blocks, L_cross = 0.

mean_coord_loss is the time average of 1 - Y / sum_b(P_b * e_b), the complement of the effective factor already inside Y.

## Agreement, discovery, revenue

Not March's exact recursion.
code starts uniform. belief starts at s.
u is the fraction of makers (S or G) whose argmax belief equals argmax(code), measured before the update. No makers => u = 0.
p2 = 0.05 + 0.2 * mean(a of makers with s·q > code·q). Empty set => the mean is 0.
vote = sum(m_i s_i) / sum(m_i) over makers.
code <- renormalize((1-p2)*code + p2*vote).
p1 = 0.02 + 0.5 * mean(k of communicators). No communicators => p1 = 0.02.
belief <- (1-p1)*belief + p1*code for every maker.

D = Y * z * (1-u). R = Y * (1-z) * u.
cum_R is entirely pre-shock. cum_D is entirely post-shock.

## Idle

Coordinator idle mass = (S_C - L)/S_C * n_C when S_C > L, else 0.
On modules, surplus used to cover L_cross is not idle.
A maker is idle if e*m < 0.5 * median(e*m among makers). The 10th percentile is not used.
iota = idle_count / N.
Y_net = Y * (1 - max(0, iota - 0.5)). D and R use Y. 0.8 is not a target.

## Promotion

At t = {PROMOTE_T}, before production, on extra cells only.
Peter: 4 makers with largest d. c <- sigmoid(beta1*d + b), then d <- 0, role <- C.
Match: 4 makers with largest b. Same update, then d <- 0, role <- C.
Tournament: no role change. Top quartile of makers by mean m over the previous 10 periods gets logit(a) += 0.15. W1-W2 = 1 is implemented as membership in that quartile, not as a random draw.

Because R is 0 at and after the shock and D is 0 before it, and t < 80 matches control, full-run paired deltas equal post-window (t>=80) paired deltas. The R gap is only t=80..99. Post-shock output gaps are in D and in post-window mean Y.

## Sweep

Modules, specialist share 0.5, pi_K = 0.1, five pi_C values, 10 seeds.
alpha in {{0.5, 1, 2}}, gamma in {{0, -0.14, -0.5}}.
beta1 in {{-1, 0, +1}} on Peter and on Match so Peter-minus-Match is defined at the same beta1.
"""
    (OUT / "DESIGN.md").write_text(text)


def slice_metrics_table(records, structures, spec, pi_K, keys):
    header = ["structure", "pi_C", "spec", "pi_K", "counts S/G/C/K"] + keys
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for st in structures:
        for pi_C in PI_C_GRID:
            r = find_rec(records, st, pi_C, spec, pi_K)
            counts = "/".join(str(c) for c in r["counts"])
            row = [st, str(pi_C), str(spec), str(pi_K), counts]
            for key in keys:
                row.append(pm(r[key]))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def full_grid_table(records):
    keys = [
        "cum_D",
        "cum_R",
        "mean_iota",
        "mean_coord_loss",
        "mean_e",
        "mean_Y",
        "mean_Y_net",
        "mean_u_pre",
        "mean_u_post",
    ]
    header = (
        ["structure", "pi_C", "spec_share", "pi_K", "counts"]
        + keys
        + ["median_t iota mean±std", "cross-seed median of median_t", "frac seeds median_t>=0.8"]
    )
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in records:
        counts = "/".join(str(c) for c in r["counts"])
        row = [r["structure"], str(r["pi_C"]), str(r["specialist_share"]), str(r["pi_K"]), counts]
        for key in keys:
            row.append(pm(r[key]))
        row.append(pm(r["median_iota"]))
        row.append(fmt(r["median_across_seeds_of_median_iota"]))
        row.append(fmt(r["frac_seeds_median_iota_ge_0.8"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_hypothesis(main_records, promotions, sweeps, beta_rows):
    sentences = []

    def cell(st, pc, ss, pk):
        return find_rec(main_records, st, pc, ss, pk)

    modules = [r for r in main_records if r["structure"] == "modules"]
    winner = sorted(
        modules,
        key=lambda r: (-r["cum_R"]["mean"], r["pi_C"], r["specialist_share"], r["pi_K"]),
    )[0]
    best_D = sorted(
        modules,
        key=lambda r: (-r["cum_D"]["mean"], r["pi_C"], r["specialist_share"], r["pi_K"]),
    )[0]
    sentences.append(
        f"On the main grid (alpha={ALPHA_MAIN}, gamma={GAMMA_MAIN}, sigma={SIGMA}, 30 seeds, ddof=1), "
        f"the modules cell with the highest mean cum_R was pi_C={winner['pi_C']}, "
        f"specialist share={winner['specialist_share']}, pi_K={winner['pi_K']} "
        f"(counts S/G/C/K={winner['counts']}) at cum_R {pm(winner['cum_R'])} and cum_D {pm(winner['cum_D'])}; "
        f"the modules cell with the highest mean cum_D was pi_C={best_D['pi_C']}, "
        f"specialist share={best_D['specialist_share']}, pi_K={best_D['pi_K']} at cum_D {pm(best_D['cum_D'])} "
        f"and cum_R {pm(best_D['cum_R'])}."
    )

    h1_bits = []
    for st in STRUCTURES:
        rows = [cell(st, pc, 0.5, 0.1) for pc in PI_C_GRID]
        lab_R, peak_R, _ = curve_label([r["cum_R"]["mean"] for r in rows])
        lab_D, peak_D, _ = curve_label([r["cum_D"]["mean"] for r in rows])
        lab_Rh, peak_Rh, iRh = curve_label([r["cum_R_highlam"]["mean"] for r in rows])
        lab_Dh, peak_Dh, iDh = curve_label([r["cum_D_highlam"]["mean"] for r in rows])
        if iDh < iRh:
            rel = "lower pi_C than high-lambda revenue"
        elif iDh == iRh:
            rel = "the same pi_C as high-lambda revenue"
        else:
            rel = "a higher pi_C than high-lambda revenue"
        vals_R = ", ".join(f"{pc}:{pm(r['cum_R'])}" for pc, r in zip(PI_C_GRID, rows))
        vals_Dh = ", ".join(f"{pc}:{pm(r['cum_D_highlam'])}" for pc, r in zip(PI_C_GRID, rows))
        vals_Rh = ", ".join(f"{pc}:{pm(r['cum_R_highlam'])}" for pc, r in zip(PI_C_GRID, rows))
        h1_bits.append(
            f"At specialist share 0.5 and pi_K=0.1 on {st}, cum_R by pi_C was [{vals_R}], shape: {lab_R}, peak pi_C={peak_R}; "
            f"cum_D shape: {lab_D}, peak pi_C={peak_D}; "
            f"high-lambda cum_R [{vals_Rh}] peaked at {peak_Rh} ({lab_Rh}); "
            f"high-lambda cum_D [{vals_Dh}] peaked at {peak_Dh}, which is {rel}."
        )
    sentences.append("Hypothesis 1: " + " ".join(h1_bits))

    flat_rows = [cell("flat", pc, 0.5, 0.1) for pc in PI_C_GRID]
    f05 = cell("flat", 0.5, 0.5, 0.1)
    f0 = cell("flat", 0.0, 0.5, 0.1)
    minR = min(flat_rows, key=lambda r: r["cum_R"]["mean"])
    minD = min(flat_rows, key=lambda r: r["cum_D"]["mean"])
    sentences.append(
        "Hypothesis 2: on flat, specialist share 0.5, pi_K=0.1, the pi_C=0.5 cell had "
        f"cum_R {pm(f05['cum_R'])}, cum_D {pm(f05['cum_D'])}, high-lambda cum_R {pm(f05['cum_R_highlam'])}, "
        f"high-lambda cum_D {pm(f05['cum_D_highlam'])}, mean_coord_loss {pm(f05['mean_coord_loss'])}, "
        f"mean_e {pm(f05['mean_e'])}, mean_iota {pm(f05['mean_iota'])}. "
        f"The lowest cum_R on that flat slice was pi_C={minR['pi_C']} at {pm(minR['cum_R'])}; "
        f"the lowest cum_D was pi_C={minD['pi_C']} at {pm(minD['cum_D'])}. "
        f"Relative to flat pi_C=0 on the same slice (cum_R {pm(f0['cum_R'])}, cum_D {pm(f0['cum_D'])}, "
        f"coord_loss {pm(f0['mean_coord_loss'])}, mean_e {pm(f0['mean_e'])}), "
        f"pi_C=0.5 {'had lower' if f05['cum_R']['mean'] < f0['cum_R']['mean'] else 'did not have lower'} cum_R and "
        f"{'had lower' if f05['cum_D']['mean'] < f0['cum_D']['mean'] else 'did not have lower'} cum_D; "
        f"coord_loss was {'higher' if f05['mean_coord_loss']['mean'] > f0['mean_coord_loss']['mean'] else 'not higher'} "
        f"and mean_e was {'lower' if f05['mean_e']['mean'] < f0['mean_e']['mean'] else 'not lower'} "
        "(lower mean_e would be more loafing). "
        "High-lambda periods are the high-lambda cumulatives above, not a separate reorganization."
    )

    n_pairs = 0
    nR = 0
    nD = 0
    for ss in SPEC_GRID:
        for pk in PI_K_GRID:
            n_pairs += 1
            mrec = cell("modules", 0.5, ss, pk)
            frec = cell("flat", 0.5, ss, pk)
            if mrec["cum_R"]["mean"] > frec["cum_R"]["mean"]:
                nR += 1
            if mrec["cum_D"]["mean"] > frec["cum_D"]["mean"]:
                nD += 1
    m05 = cell("modules", 0.5, 0.5, 0.1)
    sentences.append(
        "Hypothesis 3: at pi_C=0.5, specialist share 0.5, pi_K=0.1, modules cum_R was "
        f"{pm(m05['cum_R'])} versus flat {pm(f05['cum_R'])}, and modules cum_D was {pm(m05['cum_D'])} versus flat "
        f"{pm(f05['cum_D'])}; modules mean_coord_loss was {pm(m05['mean_coord_loss'])} versus flat "
        f"{pm(f05['mean_coord_loss'])}. Across all {n_pairs} specialist-share by pi_K pairs at pi_C=0.5, modules had higher "
        f"mean cum_R than flat in {nR}/{n_pairs} pairs and higher mean cum_D in {nD}/{n_pairs}."
    )

    h4_parts = []
    for pk in PI_K_GRID:
        n_rev = 0
        n_disc = 0
        for pc in PI_C_GRID:
            gen = cell("modules", pc, 0.25, pk)
            spe = cell("modules", pc, 0.75, pk)
            if gen["cum_R"]["mean"] > spe["cum_R"]["mean"]:
                n_rev += 1
            if spe["cum_D"]["mean"] > gen["cum_D"]["mean"]:
                n_disc += 1
        g = cell("modules", 0.125, 0.25, pk)
        s = cell("modules", 0.125, 0.75, pk)
        h4_parts.append(
            f"pi_K={pk}: generalist-heavy (share 0.25) had higher cum_R than specialist-heavy (share 0.75) "
            f"in {n_rev}/5 pi_C cells, and specialist-heavy had higher cum_D in {n_disc}/5. "
            f"At pi_C=0.125, share 0.25 cum_R {pm(g['cum_R'])}, cum_D {pm(g['cum_D'])}, "
            f"pre mean m_S {pm(g['mean_m_pre_S'])}, m_G {pm(g['mean_m_pre_G'])}, "
            f"post mean m_S {pm(g['mean_m_post_S'])}, m_G {pm(g['mean_m_post_G'])}, "
            f"init breadth kernel b*(1-H) S {pm(g['init_breadth_S'])}, G {pm(g['init_breadth_G'])}; "
            f"share 0.75 cum_R {pm(s['cum_R'])}, cum_D {pm(s['cum_D'])}, "
            f"pre m_S {pm(s['mean_m_pre_S'])}, m_G {pm(s['mean_m_pre_G'])}, "
            f"post m_S {pm(s['mean_m_post_S'])}, m_G {pm(s['mean_m_post_G'])}."
        )
    sentences.append(
        "Hypothesis 4 is read on modules, generalist-heavy = specialist share 0.25 versus specialist-heavy = 0.75. "
        "cum_R is pre-shock revenue and cum_D is post-shock discovery. " + " ".join(h4_parts)
    )

    n_up = 0
    n_down = 0
    n_all = 0
    for pc in PI_C_GRID:
        for ss in SPEC_GRID:
            n_all += 1
            hi = cell("modules", pc, ss, 0.1)
            lo = cell("modules", pc, ss, 0.0)
            if hi["cum_R"]["mean"] > lo["cum_R"]["mean"]:
                n_up += 1
            if hi["cum_D"]["mean"] < lo["cum_D"]["mean"]:
                n_down += 1
    hi = cell("modules", 0.125, 0.5, 0.1)
    lo = cell("modules", 0.125, 0.5, 0.0)
    sentences.append(
        "Hypothesis 5: on modules, pi_K=0.1 versus pi_K=0 raised mean cum_R in "
        f"{n_up}/{n_all} cells and lowered mean cum_D in {n_down}/{n_all} cells. "
        f"At pi_C=0.125 and specialist share 0.5, pi_K=0.1 had cum_R {pm(hi['cum_R'])}, cum_D {pm(hi['cum_D'])}, "
        f"mean u pre {pm(hi['mean_u_pre'])}, mean u post {pm(hi['mean_u_post'])}; "
        f"pi_K=0 had cum_R {pm(lo['cum_R'])}, cum_D {pm(lo['cum_D'])}, "
        f"mean u pre {pm(lo['mean_u_pre'])}, mean u post {pm(lo['mean_u_post'])}."
    )

    for block in promotions:
        sentences.append(
            f"Promotion at pi_C={block['pi_C']}, specialist share={block['specialist_share']}, pi_K={block['pi_K']} "
            f"on modules, beta1={block['beta1']}, {block['n_seeds']} seeds, paired treatment minus control. "
            f"Peter post-window delta_R {pm(block['peter']['delta_R_post'])}, delta_D {pm(block['peter']['delta_D_post'])}, "
            f"post mean Y {pm(block['peter']['post_mean_Y'])}; "
            f"Match delta_R {pm(block['match']['delta_R_post'])}, delta_D {pm(block['match']['delta_D_post'])}, "
            f"post mean Y {pm(block['match']['post_mean_Y'])}; "
            f"Tournament delta_R {pm(block['tournament']['delta_R_post'])}, delta_D {pm(block['tournament']['delta_D_post'])}, "
            f"post mean Y {pm(block['tournament']['post_mean_Y'])}, post mean coord_loss {pm(block['tournament']['post_mean_coord_loss'])}; "
            f"control post mean Y {pm(block['control']['post_mean_Y'])}, post mean coord_loss {pm(block['control']['post_mean_coord_loss'])}, "
            f"Peter post mean coord_loss {pm(block['peter']['post_mean_coord_loss'])}. "
            f"Paired Peter minus Match post-window R was {pm(block['peter_minus_match_post_R'])} and post-window D was "
            f"{pm(block['peter_minus_match_post_D'])}; paired Peter minus Match post-window mean Y was "
            f"{pm(block['peter_minus_match_post_Y'])}. "
            f"Paired Tournament minus Peter post-window mean Y was {pm(block['tournament_minus_peter_post_Y'])} and "
            f"post-window mean coord_loss was {pm(block['tournament_minus_peter_post_coord_loss'])}. "
            f"Full-run delta_R versus post-window delta_R for Peter: "
            f"{fmt(block['peter']['delta_R_full']['mean'])} vs {fmt(block['peter']['delta_R_post']['mean'])}; "
            f"full-run delta_D versus post-window delta_D: {fmt(block['peter']['delta_D_full']['mean'])} vs "
            f"{fmt(block['peter']['delta_D_post']['mean'])}."
        )

    n_cells = len(main_records)
    n_runs = sum(r["n_seeds"] for r in main_records)
    n_ge = 0.0
    n_cells_ge = 0
    max_med = -1.0
    max_where = None
    for r in main_records:
        n_ge += r["frac_seeds_median_iota_ge_0.8"] * r["n_seeds"]
        if r["median_across_seeds_of_median_iota"] >= 0.8:
            n_cells_ge += 1
        if r["max_seed_median_iota"] > max_med:
            max_med = r["max_seed_median_iota"]
            max_where = r
    sentences.append(
        "Hypothesis 8: iota=0.8 was not used as a target. "
        f"In {fmt(n_ge)} of {n_runs} main-grid cell-seed runs, the within-run median over periods of iota was >= 0.8. "
        f"In {n_cells_ge} of {n_cells} cells, the cross-seed median of those within-run medians was >= 0.8. "
        f"The maximum within-run median iota in any main-grid seed was {fmt(max_med)} "
        f"at structure={max_where['structure']}, pi_C={max_where['pi_C']}, specialist share={max_where['specialist_share']}, "
        f"pi_K={max_where['pi_K']} (that cell's cross-seed median of median_t iota was "
        f"{fmt(max_where['median_across_seeds_of_median_iota'])}, time-mean iota cell mean {fmt(max_where['mean_iota']['mean'])})."
    )

    groups = {}
    for row in sweeps:
        groups.setdefault((row["alpha"], row["gamma"]), []).append(row)
    default_order = rank_order(groups[(1.0, -0.14)])
    flip_desc = []
    n_diff = 0
    for key in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        order = rank_order(groups[key])
        same = order == default_order
        if not same:
            n_diff += 1
        by_pc = {r["pi_C"]: r for r in groups[key]}
        vals = ", ".join(f"{pc}:{pm(by_pc[pc]['cum_R'])}" for pc in PI_C_GRID)
        flip_desc.append(
            f"alpha={key[0]}, gamma={key[1]}: order high-to-low pi_C {order} "
            f"({'same as default' if same else 'DIFFERS from default'}); cum_R [{vals}]"
        )
    sentences.append(
        "Alpha/gamma sweep (modules, specialist share 0.5, pi_K=0.1, 10 seeds): default order of pi_C by mean cum_R "
        f"at alpha=1, gamma=-0.14 was {default_order}. {n_diff} of {len(groups)} (alpha, gamma) pairs had a different order. "
        + " ".join(flip_desc)
    )

    beta_groups = {}
    for row in beta_rows:
        beta_groups[(row["pi_C"], row["specialist_share"], row["pi_K"], row["beta1"])] = row
    pis = []
    for r in beta_rows:
        key = (r["pi_C"], r["specialist_share"], r["pi_K"])
        if key not in pis:
            pis.append(key)
    beta_bits = []
    for pi in pis:
        signs = []
        bits = []
        for b1 in (-1.0, 0.0, 1.0):
            row = beta_groups[(pi[0], pi[1], pi[2], b1)]
            diff = row["peter_minus_match_post_R"]["mean"]
            sgn = int(np.sign(diff)) if diff != 0 else 0
            signs.append(sgn)
            bits.append(
                f"beta1={b1}: Peter-Match post R {pm(row['peter_minus_match_post_R'])}, "
                f"post D {pm(row['peter_minus_match_post_D'])}"
            )
        flipped = len(set(signs)) > 1
        beta_bits.append(
            f"At pi_C={pi[0]}, specialist share={pi[1]}, pi_K={pi[2]}, 10 seeds, the sign of mean Peter-Match post-window R "
            f"{'changed' if flipped else 'did not change'} across beta1 in {{-1, 0, +1}} (signs {signs}). "
            + " ".join(bits)
            + "."
        )
    sentences.append("Beta1 sweep on Peter and Match at the same beta1: " + " ".join(beta_bits))
    return " ".join(sentences), winner, best_D


COLORS = {"modules": "#0072B2", "flat": "#D55E00", "modules_isolated": "#009E73"}


def plot_all(main_records, promotions, sweeps, beta_rows):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, key, title in (
        (axes[0], "cum_R", "Cumulative revenue (pre-shock)"),
        (axes[1], "cum_D", "Cumulative discovery (post-shock)"),
    ):
        for st in STRUCTURES:
            ys, ss = [], []
            for pc in PI_C_GRID:
                r = find_rec(main_records, st, pc, 0.5, 0.1)
                ys.append(r[key]["mean"])
                ss.append(r[key]["std"])
            ys = np.asarray(ys, dtype=float)
            ss = np.asarray(ss, dtype=float)
            ax.plot(PI_C_GRID, ys, marker="o", label=st, color=COLORS[st])
            ax.fill_between(PI_C_GRID, ys - ss, ys + ss, alpha=0.18, color=COLORS[st])
        ax.set_xlabel("pi_C")
        ax.set_ylabel(key + " (mean across seeds)")
        ax.set_title(title)
        ax.set_xticks(PI_C_GRID)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Specialist share 0.5, pi_K=0.1, 30 seeds; band is ±1 seed-level std")
    p = PLOT_DIR / "cum_RD_vs_piC_by_structure.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    mats = []
    for pk in PI_K_GRID:
        mat = np.zeros((len(SPEC_GRID), len(PI_C_GRID)))
        for i, ss in enumerate(SPEC_GRID):
            for j, pc in enumerate(PI_C_GRID):
                mat[i, j] = find_rec(main_records, "modules", pc, ss, pk)["cum_R"]["mean"]
        mats.append(mat)
    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    im = None
    for ax, mat, pk in zip(axes, mats, PI_K_GRID):
        im = ax.imshow(mat, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(PI_C_GRID)), [str(x) for x in PI_C_GRID])
        ax.set_yticks(range(len(SPEC_GRID)), [str(x) for x in SPEC_GRID])
        ax.set_xlabel("pi_C")
        ax.set_ylabel("specialist share")
        ax.set_title(f"modules cum_R, pi_K={pk}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="mean cum_R")
    p = PLOT_DIR / "heatmap_cumR_modules.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    max_med = -1.0
    for st, color in (("flat", "#D55E00"), ("modules", "#0072B2")):
        ys, ss, meds = [], [], []
        for pc in PI_C_GRID:
            r = find_rec(main_records, st, pc, 0.5, 0.1)
            ys.append(r["mean_iota"]["mean"])
            ss.append(r["mean_iota"]["std"])
            meds.append(r["median_across_seeds_of_median_iota"])
            max_med = max(max_med, r["max_seed_median_iota"])
        ys = np.asarray(ys, dtype=float)
        ss = np.asarray(ss, dtype=float)
        ax.plot(PI_C_GRID, ys, marker="o", color=color, label=f"{st} time-mean iota")
        ax.fill_between(PI_C_GRID, ys - ss, ys + ss, alpha=0.18, color=color)
        ax.plot(
            PI_C_GRID,
            meds,
            marker="x",
            linestyle="--",
            color=color,
            label=f"{st} cross-seed median of median_t iota",
        )
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="0.8 reference (not a target)")
    ax.set_xlabel("pi_C")
    ax.set_ylabel("iota")
    ax.set_xticks(PI_C_GRID)
    ax.set_title(f"Idle fraction, spec 0.5, pi_K=0.1. Max seed median_t iota on these lines: {max_med:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    p = PLOT_DIR / "iota_vs_piC_flat_vs_modules.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    width = 0.18
    x = np.arange(len(PI_C_GRID))
    for ax, pk in zip(axes, PI_K_GRID):
        series = [
            ("gen cum_R", 0.25, "cum_R", "#56B4E9"),
            ("spec cum_R", 0.75, "cum_R", "#0072B2"),
            ("gen cum_D", 0.25, "cum_D", "#E69F00"),
            ("spec cum_D", 0.75, "cum_D", "#D55E00"),
        ]
        for s_i, (lab, ss, key, color) in enumerate(series):
            means, stds = [], []
            for pc in PI_C_GRID:
                r = find_rec(main_records, "modules", pc, ss, pk)
                means.append(r[key]["mean"])
                stds.append(r[key]["std"])
            ax.bar(x + (s_i - 1.5) * width, means, width=width, yerr=stds, label=lab, color=color, capsize=2)
        ax.set_xticks(x, [str(pc) for pc in PI_C_GRID])
        ax.set_xlabel("pi_C")
        ax.set_title(f"modules, pi_K={pk}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("mean cumulative (seed std bars)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Generalist-heavy (share 0.25) vs specialist-heavy (share 0.75)")
    p = PLOT_DIR / "preshock_postshock_spec_vs_gen.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    n_blocks = len(promotions)
    fig, axes = plt.subplots(1, n_blocks, figsize=(5.2 * n_blocks, 4.3), constrained_layout=True, squeeze=False)
    treatments = ["control", "peter", "match", "tournament"]
    for ax, block in zip(axes[0], promotions):
        xx = np.arange(len(treatments))
        dR = [block[t]["delta_R_post"]["mean"] for t in treatments]
        sR = [block[t]["delta_R_post"]["std"] for t in treatments]
        dD = [block[t]["delta_D_post"]["mean"] for t in treatments]
        sD = [block[t]["delta_D_post"]["std"] for t in treatments]
        ax.bar(xx - 0.18, dR, width=0.36, yerr=sR, capsize=2, label="delta R post window", color="#0072B2")
        ax.bar(xx + 0.18, dD, width=0.36, yerr=sD, capsize=2, label="delta D post window", color="#D55E00")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(xx, treatments)
        ax.set_title(
            f"pi_C={block['pi_C']}, share={block['specialist_share']}, pi_K={block['pi_K']}, beta1={block['beta1']}"
        )
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel("paired delta vs control (mean ± seed std)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Promotion at t=80. Post-window R only moves on t=80..99 because R=0 after the shock.")
    p = PLOT_DIR / "promotion_deltas.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True, sharey=True)
    gammas = [0.0, -0.14, -0.5]
    alphas = [0.5, 1.0, 2.0]
    colors = {0.5: "#009E73", 1.0: "#0072B2", 2.0: "#D55E00"}
    by = {(r["alpha"], r["gamma"], r["pi_C"]): r for r in sweeps}
    for ax, g in zip(axes, gammas):
        for a in alphas:
            ys = np.array([by[(a, g, pc)]["cum_R"]["mean"] for pc in PI_C_GRID], dtype=float)
            ss = np.array([by[(a, g, pc)]["cum_R"]["std"] for pc in PI_C_GRID], dtype=float)
            ax.plot(PI_C_GRID, ys, marker="o", color=colors[a], label=f"alpha={a}")
            ax.fill_between(PI_C_GRID, ys - ss, ys + ss, color=colors[a], alpha=0.15)
        ax.set_title(f"gamma={g}")
        ax.set_xlabel("pi_C")
        ax.set_xticks(PI_C_GRID)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("cum_R")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sweep: modules, specialist share 0.5, pi_K=0.1, 10 seeds, ±1 std")
    p = PLOT_DIR / "sweep_alpha_gamma_cumR.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    pis = []
    for r in beta_rows:
        key = (r["pi_C"], r["specialist_share"], r["pi_K"])
        if key not in pis:
            pis.append(key)
    fig, axes = plt.subplots(1, len(pis), figsize=(5.4 * len(pis), 4.0), constrained_layout=True, squeeze=False)
    b1s = [-1.0, 0.0, 1.0]
    byb = {(r["pi_C"], r["specialist_share"], r["pi_K"], r["beta1"]): r for r in beta_rows}
    for ax, pi in zip(axes[0], pis):
        xx = np.arange(len(b1s))
        dR = [byb[(pi[0], pi[1], pi[2], b)]["peter_minus_match_post_R"]["mean"] for b in b1s]
        sR = [byb[(pi[0], pi[1], pi[2], b)]["peter_minus_match_post_R"]["std"] for b in b1s]
        dD = [byb[(pi[0], pi[1], pi[2], b)]["peter_minus_match_post_D"]["mean"] for b in b1s]
        sD = [byb[(pi[0], pi[1], pi[2], b)]["peter_minus_match_post_D"]["std"] for b in b1s]
        ax.bar(xx - 0.18, dR, width=0.36, yerr=sR, capsize=2, label="Peter-Match post R", color="#0072B2")
        ax.bar(xx + 0.18, dD, width=0.36, yerr=sD, capsize=2, label="Peter-Match post D", color="#D55E00")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(xx, [str(b) for b in b1s])
        ax.set_xlabel("beta1")
        ax.set_title(f"pi_C={pi[0]}, share={pi[1]}, pi_K={pi[2]}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel("paired difference (10 seeds)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Beta1 sweep: Peter minus Match on the post-promotion window")
    p = PLOT_DIR / "sweep_beta1_peter_match.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))
    return paths


def promotion_block(pop, tasks, cfg, counts, beta1, n_seeds):
    seed_index = np.arange(n_seeds)
    raw = {}
    for kind in ("control", "peter", "match", "tournament"):
        out = run_from_pop(
            pop,
            tasks,
            seed_index,
            alpha=ALPHA_MAIN,
            gamma=GAMMA_MAIN,
            structures=["modules"],
            promotion=None if kind == "control" else kind,
            beta1=beta1,
            link_mode="brooks",
        )
        raw[kind] = out["modules"]
    control = raw["control"]
    summaries = {}
    for kind in ("control", "peter", "match", "tournament"):
        rec = summarize_run(raw[kind])
        post_D = raw[kind]["post_D"] - control["post_D"]
        full_D = raw[kind]["cum_D"] - control["cum_D"]
        post_R = raw[kind]["post_R"] - control["post_R"]
        full_R = raw[kind]["cum_R"] - control["cum_R"]
        if not np.allclose(post_D, full_D, atol=1e-6):
            raise RuntimeError(f"D delta mismatch {kind}")
        if not np.allclose(post_R, full_R, atol=1e-6):
            raise RuntimeError(f"R delta mismatch {kind}")
        rec["delta_D_post"] = pack_strict(post_D)
        rec["delta_D_full"] = pack_strict(full_D)
        rec["delta_R_post"] = pack_strict(post_R)
        rec["delta_R_full"] = pack_strict(full_R)
        summaries[kind] = rec
    return {
        "pi_C": cfg["pi_C"],
        "specialist_share": cfg["specialist_share"],
        "pi_K": cfg["pi_K"],
        "counts": [int(c) for c in counts],
        "structure": "modules",
        "beta1": beta1,
        "n_seeds": n_seeds,
        "control": summaries["control"],
        "peter": summaries["peter"],
        "match": summaries["match"],
        "tournament": summaries["tournament"],
        "peter_minus_match_post_R": pack_strict(raw["peter"]["post_R"] - raw["match"]["post_R"]),
        "peter_minus_match_post_D": pack_strict(raw["peter"]["post_D"] - raw["match"]["post_D"]),
        "peter_minus_match_post_Y": pack_strict(raw["peter"]["post_mean_Y"] - raw["match"]["post_mean_Y"]),
        "tournament_minus_peter_post_Y": pack_strict(raw["tournament"]["post_mean_Y"] - raw["peter"]["post_mean_Y"]),
        "tournament_minus_peter_post_coord_loss": pack_strict(
            raw["tournament"]["post_mean_coord_loss"] - raw["peter"]["post_mean_coord_loss"]
        ),
    }


def beta_pair(pop, tasks, cfg, n_seeds, beta1):
    seed_index = np.arange(n_seeds)
    raw = {}
    for kind in ("control", "peter", "match"):
        out = run_from_pop(
            pop,
            tasks,
            seed_index,
            alpha=ALPHA_MAIN,
            gamma=GAMMA_MAIN,
            structures=["modules"],
            promotion=None if kind == "control" else kind,
            beta1=beta1,
            link_mode="brooks",
        )
        raw[kind] = out["modules"]
    return {
        "pi_C": cfg["pi_C"],
        "specialist_share": cfg["specialist_share"],
        "pi_K": cfg["pi_K"],
        "beta1": beta1,
        "n_seeds": n_seeds,
        "structure": "modules",
        "peter_minus_match_post_R": pack_strict(raw["peter"]["post_R"] - raw["match"]["post_R"]),
        "peter_minus_match_post_D": pack_strict(raw["peter"]["post_D"] - raw["match"]["post_D"]),
    }


def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


def write_results_md(payload, paragraph):
    main_records = payload["main_grid"]
    lines = []
    lines.append("# Results")
    lines.append("")
    lines.append(
        "Every number below is taken from results.json for this run. Means are across seeds. "
        "The ± quantity is the seed-level sample standard deviation (ddof=1). "
        "cum_R is pre-shock revenue because R_t = 0 for t >= 100. "
        "cum_D is post-shock discovery because D_t = 0 for t < 100. "
        "Primary D and R use Y, not Y_net."
    )
    lines.append("")
    lines.append("## Main slice used in the structure plot")
    lines.append("")
    lines.append("Specialist share 0.5, pi_K = 0.1. High-lambda columns sum only periods with lambda_t = 0.8.")
    lines.append("")
    lines.append(
        slice_metrics_table(
            main_records,
            STRUCTURES,
            0.5,
            0.1,
            ["cum_D", "cum_R", "cum_D_highlam", "cum_R_highlam", "mean_iota", "mean_coord_loss", "mean_e"],
        )
    )
    lines.append("")
    lines.append("## Full main grid")
    lines.append("")
    lines.append(full_grid_table(main_records))
    lines.append("")
    lines.append("## Promotion")
    lines.append("")
    lines.append(
        "Paired deltas are treatment minus no-promotion control, same seeds. "
        "Post window is t=80..199. Full-run deltas matched post-window deltas in this run."
    )
    lines.append("")
    for block in payload["promotion"]:
        lines.append(
            f"### pi_C={block['pi_C']}, specialist share={block['specialist_share']}, "
            f"pi_K={block['pi_K']}, beta1={block['beta1']}, n_seeds={block['n_seeds']}"
        )
        lines.append("")
        lines.append(
            "| treatment | cum_D | cum_R | delta_D post | delta_R post | delta_D full | delta_R full | "
            "post mean Y | post mean coord loss | post mean e |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for kind in ("control", "peter", "match", "tournament"):
            r = block[kind]
            lines.append(
                "| "
                + " | ".join(
                    [
                        kind,
                        pm(r["cum_D"]),
                        pm(r["cum_R"]),
                        pm(r["delta_D_post"]),
                        pm(r["delta_R_post"]),
                        pm(r["delta_D_full"]),
                        pm(r["delta_R_full"]),
                        pm(r["post_mean_Y"]),
                        pm(r["post_mean_coord_loss"]),
                        pm(r["post_mean_e"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(
            f"Peter minus Match post R: {pm(block['peter_minus_match_post_R'])}. "
            f"Peter minus Match post D: {pm(block['peter_minus_match_post_D'])}. "
            f"Peter minus Match post mean Y: {pm(block['peter_minus_match_post_Y'])}. "
            f"Tournament minus Peter post mean Y: {pm(block['tournament_minus_peter_post_Y'])}. "
            f"Tournament minus Peter post mean coord loss: {pm(block['tournament_minus_peter_post_coord_loss'])}."
        )
        lines.append("")
    lines.append("## Alpha / gamma sweep")
    lines.append("")
    lines.append("Modules, specialist share 0.5, pi_K=0.1, 10 seeds. cum_R mean ± std.")
    lines.append("")
    lines.append("| alpha | gamma | " + " | ".join(f"pi_C={pc}" for pc in PI_C_GRID) + " |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(PI_C_GRID)) + " |")
    groups = {}
    for row in payload["sweep_alpha_gamma"]:
        groups.setdefault((row["alpha"], row["gamma"]), {})[row["pi_C"]] = row
    for key in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        bypc = groups[key]
        lines.append(
            "| " + " | ".join([str(key[0]), str(key[1])] + [pm(bypc[pc]["cum_R"]) for pc in PI_C_GRID]) + " |"
        )
    lines.append("")
    lines.append("## Beta1 sweep (Peter and Match)")
    lines.append("")
    lines.append("| pi_C | spec | pi_K | beta1 | Peter-Match post R | Peter-Match post D |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["sweep_beta1"]:
        lines.append(
            f"| {row['pi_C']} | {row['specialist_share']} | {row['pi_K']} | {row['beta1']} | "
            f"{pm(row['peter_minus_match_post_R'])} | {pm(row['peter_minus_match_post_D'])} |"
        )
    lines.append("")
    if payload.get("sensitivity_graicunas"):
        lines.append("## Sensitivity (Graicunas) — not part of the main grid")
        lines.append("")
        lines.append(
            "L = lambda_t * load(n) with load(n) = n*(2**(n-1)+n-1), modules, specialist share 0.5, "
            "pi_K=0.1, 10 seeds, alpha=1, gamma=-0.14. Not used to score the hypotheses."
        )
        lines.append("")
        lines.append("| pi_C | cum_D | cum_R | mean_coord_loss | mean_e |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in payload["sensitivity_graicunas"]:
            lines.append(
                f"| {row['pi_C']} | {pm(row['cum_D'])} | {pm(row['cum_R'])} | "
                f"{pm(row['mean_coord_loss'])} | {pm(row['mean_e'])} |"
            )
        lines.append("")
    lines.append("## Hypothesis note")
    lines.append("")
    lines.append(paragraph)
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for pth in payload.get("plots", []):
        lines.append(f"- `{pth}`")
    lines.append("")
    (OUT / "results.md").write_text("\n".join(lines))


def smoke():
    print("smoke", flush=True)
    for pi_C in PI_C_GRID:
        for spec in SPEC_GRID:
            for pi_K in PI_K_GRID:
                pi = make_pi(pi_C, spec, pi_K)
                counts = largest_remainder(pi, N_AGENTS)
                assert int(counts.sum()) == N_AGENTS
                if np.max(np.abs(counts - pi * N_AGENTS)) >= 1.0 + 1e-6:
                    raise RuntimeError((pi, counts))
    pi = make_pi(0.125, 0.5, 0.1)
    pop = build_pop([0, 1], pi)
    role0 = pop["role"].copy()
    theta0 = pop["theta"].copy()
    m_recent = np.random.default_rng(0).random(role0.shape)
    role = role0.copy()
    theta = theta0.copy()
    k = apply_promotion("peter", role, theta, m_recent, beta1=-1.0)
    assert k == 4
    new_c = (role == 2) & (role0 != 2)
    assert int(new_c[0].sum()) == 4
    assert np.all(theta[0, new_c[0], 0] == 0.0)
    d_old = theta0[0, new_c[0], 0]
    b_old = theta0[0, new_c[0], 1]
    expect = sigmoid(-1.0 * d_old + b_old)
    assert np.allclose(theta[0, new_c[0], 2], expect)
    role_t = role0.copy()
    theta_t = theta0.copy()
    apply_promotion("tournament", role_t, theta_t, m_recent, beta1=-1.0)
    assert np.array_equal(role_t, role0)
    seeds = [0, 1]
    tasks = build_tasks(seeds, T=30, shock_t=15)
    pop = build_pop(seeds, pi)
    out = run_from_pop(
        pop,
        tasks,
        np.arange(2),
        alpha=1.0,
        gamma=-0.14,
        structures=STRUCTURES,
        promotion=None,
        promote_t=12,
    )
    for metrics in out.values():
        assert np.all(metrics["cum_D"] >= -1e-8)
        assert np.all(metrics["cum_R"] >= -1e-8)
    out_p = run_from_pop(
        pop,
        tasks,
        np.arange(2),
        alpha=1.0,
        gamma=-0.14,
        structures=["modules"],
        promotion="peter",
        beta1=-1.0,
        promote_t=12,
    )
    assert out_p["modules"]["_promoted_k"] == 4
    H = pop["H"]
    b = pop["theta"][:, :, 1]
    d = pop["theta"][:, :, 0]
    for r, name in enumerate(ROLE_NAME):
        mask = pop["role"] == r
        if not np.any(mask):
            continue
        print(
            f"  role {name}: n={int(mask[0].sum())} meanH={float(H[mask].mean()):.3f} "
            f"breadth={float((b * (1 - H))[mask].mean()):.3f} d={float(d[mask].mean()):.3f}",
            flush=True,
        )
    print("smoke ok", flush=True)


def full():
    t0 = time.time()
    write_readme()
    seeds = list(range(N_SEEDS_MAIN))
    print("building tasks", flush=True)
    tasks = build_tasks(seeds, T=T_MAIN, shock_t=SHOCK_T)
    configs = base_configs()
    print(f"building {len(configs)} populations", flush=True)
    pops = {}
    counts = {}
    for cfg in configs:
        key = cfg_key(cfg["pi_C"], cfg["specialist_share"], cfg["pi_K"])
        pop = build_pop(seeds, cfg["pi"], sigma=SIGMA)
        pops[key] = pop
        counts[key] = pop["counts"]
        print(f"  counts {key}: {pop['counts'].tolist()}", flush=True)

    print("main grid", flush=True)
    main_records = []
    for i, cfg in enumerate(configs):
        key = cfg_key(cfg["pi_C"], cfg["specialist_share"], cfg["pi_K"])
        out = run_from_pop(
            pops[key],
            tasks,
            np.arange(N_SEEDS_MAIN),
            alpha=ALPHA_MAIN,
            gamma=GAMMA_MAIN,
            structures=STRUCTURES,
            promotion=None,
            link_mode="brooks",
        )
        for st in STRUCTURES:
            rec = summarize_run(out[st])
            rec = attach_identity(rec, cfg, st, N_SEEDS_MAIN, counts[key])
            main_records.append(rec)
        print(
            f"  cell {i+1}/{len(configs)} {key} elapsed {time.time()-t0:.1f}s "
            f"modules cum_R {out['modules']['cum_R'].mean():.4f}",
            flush=True,
        )

    modules = [r for r in main_records if r["structure"] == "modules"]
    winner = sorted(
        modules,
        key=lambda r: (-r["cum_R"]["mean"], r["pi_C"], r["specialist_share"], r["pi_K"]),
    )[0]
    print(
        "winner",
        winner["pi_C"],
        winner["specialist_share"],
        winner["pi_K"],
        winner["cum_R"]["mean"],
        flush=True,
    )
    ref_pi = (0.125, 0.5, 0.1)
    win_pi = (winner["pi_C"], winner["specialist_share"], winner["pi_K"])
    prom_pis = [win_pi]
    if ref_pi != win_pi:
        prom_pis.append(ref_pi)

    print("promotion", prom_pis, flush=True)
    promotions = []
    for pi_C, spec, pi_K in prom_pis:
        cfg = {
            "pi_C": pi_C,
            "specialist_share": spec,
            "pi_K": pi_K,
            "pi": make_pi(pi_C, spec, pi_K),
        }
        key = cfg_key(pi_C, spec, pi_K)
        block = promotion_block(pops[key], tasks, cfg, counts[key], BETA1_MAIN, N_SEEDS_MAIN)
        promotions.append(block)
        print(
            f"  promotion {key} peter-match R {block['peter_minus_match_post_R']['mean']:.4f}",
            flush=True,
        )

    print("alpha/gamma sweep", flush=True)
    sweeps = []
    sweep_seed_index = np.arange(N_SEEDS_SWEEP)
    for alpha in (0.5, 1.0, 2.0):
        for gamma in (0.0, -0.14, -0.5):
            for pi_C in PI_C_GRID:
                cfg = {
                    "pi_C": pi_C,
                    "specialist_share": 0.5,
                    "pi_K": 0.1,
                    "pi": make_pi(pi_C, 0.5, 0.1),
                }
                key = cfg_key(pi_C, 0.5, 0.1)
                out = run_from_pop(
                    pops[key],
                    tasks,
                    sweep_seed_index,
                    alpha=alpha,
                    gamma=gamma,
                    structures=["modules"],
                    promotion=None,
                    link_mode="brooks",
                )
                rec = summarize_run(out["modules"])
                rec = attach_identity(
                    rec,
                    cfg,
                    "modules",
                    N_SEEDS_SWEEP,
                    counts[key],
                    extra={"alpha": alpha, "gamma": gamma},
                )
                sweeps.append(rec)
            print(f"  sweep alpha={alpha} gamma={gamma} elapsed {time.time()-t0:.1f}s", flush=True)

    print("beta1 sweep", flush=True)
    beta_rows = []
    for pi_C, spec, pi_K in prom_pis:
        cfg = {
            "pi_C": pi_C,
            "specialist_share": spec,
            "pi_K": pi_K,
            "pi": make_pi(pi_C, spec, pi_K),
        }
        key = cfg_key(pi_C, spec, pi_K)
        for beta1 in (-1.0, 0.0, 1.0):
            row = beta_pair(pops[key], tasks, cfg, N_SEEDS_SWEEP, beta1)
            beta_rows.append(row)
            print(
                f"  beta {key} beta1={beta1} P-M R {row['peter_minus_match_post_R']['mean']:.4f}",
                flush=True,
            )

    sensitivity = []
    if time.time() - t0 < 420:
        print("graicunas sensitivity", flush=True)
        for pi_C in PI_C_GRID:
            cfg = {
                "pi_C": pi_C,
                "specialist_share": 0.5,
                "pi_K": 0.1,
                "pi": make_pi(pi_C, 0.5, 0.1),
            }
            key = cfg_key(pi_C, 0.5, 0.1)
            out = run_from_pop(
                pops[key],
                tasks,
                sweep_seed_index,
                alpha=ALPHA_MAIN,
                gamma=GAMMA_MAIN,
                structures=["modules"],
                promotion=None,
                link_mode="graicunas",
            )
            rec = summarize_run(out["modules"])
            rec = attach_identity(
                rec,
                cfg,
                "modules",
                N_SEEDS_SWEEP,
                counts[key],
                extra={"link_mode": "graicunas", "label": "sensitivity"},
            )
            sensitivity.append(rec)
        print(f"  sensitivity done {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"skipping graicunas; elapsed {time.time()-t0:.1f}s", flush=True)

    paragraph, winner, best_D = build_hypothesis(main_records, promotions, sweeps, beta_rows)
    payload = {
        "config": {
            "N": N_AGENTS,
            "K": K_DOM,
            "T": T_MAIN,
            "shock_t": SHOCK_T,
            "promote_t": PROMOTE_T,
            "n_seeds_main": N_SEEDS_MAIN,
            "n_seeds_sweep": N_SEEDS_SWEEP,
            "sigma": SIGMA,
            "alpha_main": ALPHA_MAIN,
            "gamma_main": GAMMA_MAIN,
            "eta_C": ETA_C,
            "eta_K": ETA_K,
            "delta_idle": DELTA_IDLE,
            "iota0": IOTA0,
            "cross_lambda": CROSS_LAMBDA,
            "cross_pairs": CROSS_PAIRS,
            "cross_load": CROSS_LOAD,
            "std": "sample std across seeds, ddof=1",
            "z_weight": "0 before shock (breadth only), 1 at and after shock (depth only)",
            "link_mode_main": "brooks",
            "sparse_fraction_realized": float(tasks["sparse"].mean()),
            "n_low_lambda_periods": int(np.sum(tasks["lam"] < 0.5)),
            "n_high_lambda_periods": int(np.sum(tasks["lam"] >= 0.5)),
            "winner_modules_cum_R": {
                "pi_C": winner["pi_C"],
                "specialist_share": winner["specialist_share"],
                "pi_K": winner["pi_K"],
                "cum_R_mean": winner["cum_R"]["mean"],
                "cum_R_std": winner["cum_R"]["std"],
            },
            "best_modules_cum_D": {
                "pi_C": best_D["pi_C"],
                "specialist_share": best_D["specialist_share"],
                "pi_K": best_D["pi_K"],
                "cum_D_mean": best_D["cum_D"]["mean"],
                "cum_D_std": best_D["cum_D"]["std"],
            },
            "elapsed_sec_before_plots": time.time() - t0,
        },
        "main_grid": main_records,
        "promotion": promotions,
        "sweep_alpha_gamma": sweeps,
        "sweep_beta1": beta_rows,
        "sensitivity_graicunas": sensitivity,
        "hypothesis_paragraph": paragraph,
        "plots": [],
    }
    (OUT / "results.json").write_text(json.dumps(json_clean(payload), indent=2))
    write_results_md(payload, paragraph)
    print("writing plots", flush=True)
    paths = plot_all(main_records, promotions, sweeps, beta_rows)
    payload["plots"] = paths
    payload["config"]["elapsed_sec"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(json_clean(payload), indent=2))
    write_results_md(payload, paragraph)
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        smoke()
    else:
        smoke()
        full()
