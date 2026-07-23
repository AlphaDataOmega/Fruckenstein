"""UDI organ 4: the voxel field — converts distributed learned relational
evidence into spatial pressure. Every equation here is declared in the
lineage protocol §voxel/§diffusion/§readout. No route, goal,
or body bias; no action->voxel, action->direction, funnel->candidate, or
joint->direction assignment. All ratios logged; pressure exactly
reconstructable offline from the logs."""
import itertools

import numpy as np

from constants import (
    VOX_SHAPE, NULL_BASE, KERNEL_POWER, DIFFUSION_LAMBDA, DIFFUSION_ITER,
    BELIEF_BETA, CHOICE_ETA,
)

# lattice positions in {-1,0,1}^3 (seat frame), fixed order
POSITIONS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
N_VOX = len(POSITIONS)  # 27

# the 12 algorithmic proposal voxels: exactly two nonzero coordinates
ALGO_MASK = np.array([np.count_nonzero(p) == 2 for p in POSITIONS])
ALGO_IDX = np.where(ALGO_MASK)[0]
assert len(ALGO_IDX) == 12

# unit positions for algorithmic voxels
UNIT_POS = np.zeros_like(POSITIONS)
_norms = np.linalg.norm(POSITIONS, axis=1)
UNIT_POS[_norms > 0] = POSITIONS[_norms > 0] / _norms[_norms > 0, None]

# the 8 declared local spatial directions: cube-corner units
DIRECTIONS = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) / np.sqrt(3.0)
N_DIR = 8
E_NULL = np.zeros(N_DIR + 1)
E_NULL[N_DIR] = 1.0  # pure-null ratio

# 6-face adjacency
_NEIGHBORS = []
for i, p in enumerate(POSITIONS):
    ns = []
    for j, q in enumerate(POSITIONS):
        if np.abs(p - q).sum() == 1.0:
            ns.append(j)
    _NEIGHBORS.append(np.array(ns, dtype=int))

# neighbor-mean operator as a matrix: row v holds 1/deg(v) at v's neighbors
# (identical arithmetic to the per-voxel mean, vectorized)
_W_MEAN = np.zeros((N_VOX, N_VOX))
for _v, _ns in enumerate(_NEIGHBORS):
    _W_MEAN[_v, _ns] = 1.0 / len(_ns)


class VoxelField:
    """One decision-cycle field: propose -> diffuse -> read out."""

    def __init__(self, direction_perm=None, null_disabled=False):
        # condition F: permute the direction vectors used in the READOUT only
        self.readout_dirs = (DIRECTIONS if direction_perm is None
                             else DIRECTIONS[direction_perm])
        self.null_disabled = null_disabled
        self.last_proposals = None
        self.last_diffused = None
        self.last_convergence = None
        # persistent believed field B_t (candidate 2): distributed inferred
        # state, integrated from observation-driven proposals each tick,
        # reorganized by the backward choice-difference pass, reset only at
        # lifecycle events. The ONLY write target of choice feedback.
        self.B = np.tile(E_NULL, (N_VOX, 1))

    def reset_believed(self):
        """Lifecycle event: believed field returns to pure null."""
        self.B = np.tile(E_NULL, (N_VOX, 1))

    def integrate(self, proposals):
        """Forward believed-state update (candidate 2, declared):
        B <- normalize((1-beta)*B + beta*P_t), then diffusion on B.
        Returns the post-diffusion believed field used for readout."""
        b = (1 - BELIEF_BETA) * self.B + BELIEF_BETA * proposals
        sums = b.sum(axis=1, keepdims=True)
        b = np.where(sums > 1e-12, b / sums, E_NULL)
        self.B = self.diffuse(b)
        return self.B

    def leak_toward_null(self, lam):
        """Candidate 4 believed-field law: at a decision boundary in an
        unsupported/off-model state, the believed field leaks toward null
        by lam in [0,1]. Prior commitment may not create confidence in a
        newly encountered band."""
        lam = float(np.clip(lam, 0.0, 1.0))
        if lam <= 0:
            return
        b = (1 - lam) * self.B + lam * E_NULL[None, :]
        sums = b.sum(axis=1, keepdims=True)
        self.B = np.where(sums > 1e-12, b / sums, E_NULL)

    def choice_feedback(self, disposition, eta_scale=1.0):
        """Backward pass (candidate 2, declared): the resolved-choice ratio
        pattern pulls the believed field, eta = CHOICE_ETA. Targets B ONLY —
        no ledger, no observation, no world state is reachable from here.
        The commitment becomes durable capability evidence only through the
        consequence channel at bout end."""
        dp = np.asarray(disposition, dtype=float)
        raw = np.maximum(0.0, DIRECTIONS @ dp)
        s = raw.sum()
        if s <= 1e-12:
            return
        q_star = np.zeros(N_DIR + 1)
        q_star[:N_DIR] = raw / s  # null component of the commitment is zero
        b = self.B + CHOICE_ETA * float(np.clip(eta_scale, 0.0, 1.0)) \
            * (q_star[None, :] - self.B)
        sums = b.sum(axis=1, keepdims=True)
        self.B = np.where(sums > 1e-12, b / sums, E_NULL)

    def propose(self, weights, dirs, reliabilities, null_weights=None):
        """Algorithmic proposal equation (protocol, exact).
        weights: w_a per token (trio or body-only evidence weight);
        dirs: learned effect unit vectors d_a (None where unavailable);
        reliabilities: r_a per token.
        null_weights (candidate 4, support-coupled null): explicit per-token
        null mass (1-rho)*|adv|*scale — replaces the legacy unreliability
        term so ignorance reaches the field as null, never as direction."""
        q = np.tile(E_NULL, (N_VOX, 1))  # passive voxels: pure null
        avail = [(w, d, r, (null_weights[i] if null_weights is not None else None))
                 for i, (w, d, r) in enumerate(zip(weights, dirs, reliabilities))
                 if d is not None]
        r_bar = np.mean([r for _w, _d, r, _nw in avail]) if avail else 1.0
        for v in ALGO_IDX:
            xv = UNIT_POS[v]
            e_v = np.zeros(3)
            null_raw = NULL_BASE
            for w, d, r, nw in avail:
                kappa = max(0.0, float(xv @ d)) ** KERNEL_POWER
                e_v += kappa * w * d
                if nw is not None:
                    null_raw += kappa * nw
                else:
                    null_raw += kappa * (1.0 - r) * abs(w) / max(r_bar, 1e-9)
            raw = np.empty(N_DIR + 1)
            raw[:N_DIR] = np.maximum(0.0, DIRECTIONS @ e_v)
            raw[N_DIR] = 0.0 if self.null_disabled else null_raw
            s = raw.sum()
            q[v] = raw / s if s > 1e-12 else E_NULL.copy()
        if self.null_disabled:
            # renormalize pure-null passives to uniform directional mass
            for v in range(N_VOX):
                if not ALGO_MASK[v]:
                    q[v] = np.full(N_DIR + 1, 0.0)
                    q[v][:N_DIR] = 1.0 / N_DIR
        self.last_proposals = q.copy()
        return q

    def diffuse(self, q):
        """Synchronous (Jacobi) diffusion, protocol-exact:
        q_v <- (1-l)q_v + l*mean(face neighbors); renormalize each iteration.
        No wrap, no ghost mass; null diffuses like any component."""
        conv = None
        for _ in range(DIFFUSION_ITER):
            nxt = (1 - DIFFUSION_LAMBDA) * q + DIFFUSION_LAMBDA * (_W_MEAN @ q)
            sums = nxt.sum(axis=1, keepdims=True)
            nxt = np.where(sums > 1e-12, nxt / sums, E_NULL)
            conv = float(np.abs(nxt - q).sum(axis=1).mean())
            q = nxt
        self.last_diffused = q.copy()
        self.last_convergence = conv
        return q

    def readout(self, q):
        """Field readout, protocol-exact: p_v = mass_v * sum_d q_v[d]*u_d;
        pressure = (1/27) * sum_v p_v (unscaled; S_udi applied by caller).
        Null contributes no outward vector."""
        return (q[:, :N_DIR] @ self.readout_dirs).mean(axis=0)

    def metrics(self, q):
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(q > 1e-12, q * np.log(q), 0.0).sum(axis=1)
        return {
            "nullMassMean": float(q[:, N_DIR].mean()),
            "nullMassAlgo": float(q[ALGO_IDX, N_DIR].mean()),
            "entropyMean": float(ent.mean()),
            "convergenceL1": self.last_convergence,
        }


def reconstruct_pressure(diffused_ratios, direction_perm=None):
    """Offline reconstruction (gate S1-G1): pressure from logged ratios."""
    dirs = DIRECTIONS if direction_perm is None else DIRECTIONS[direction_perm]
    q = np.asarray(diffused_ratios)
    return (q[:, :N_DIR] @ dirs).mean(axis=0)
