"""UDI core constants — frozen laws carried by value, new organ constants
declared. NO body, task, or semantic content. Every constant here appears
in the lineage protocol document one directory up."""

# ---- Phase 14 relational valence / prospect laws (verbatim lineage) ----
CONTEXT_COUNT = 1024
THETA_LOW = 0.35
THETA_REARM = 0.55
CTX_AGE_CAP = 6000                       # H = attempt_ticks / 1.25
CTX_DECAY = 0.08 ** (1.0 / CTX_AGE_CAP)
CTX_FLOOR = 0.08
CTX_REFRESH_WINDOW = 30
CTX_PER_CONTEXT_CAP = 3
CTX_GLOBAL_CAP = 512
ACT_ELIG_DECAY = 0.985
ACT_ELIG_AGE = 120
ACT_ELIG_FLOOR = 0.08
ACT_ELIG_CAP = 96
EV_COH_CAP = 0.9
EV_COH_F = 0.94
EV_ADV_F = 0.82
EV_ADV_MIN = 0.05
REV_SCALE = 0.46
REV_CAP = 4
BEN_CAP = 0.62
BEN_F = 0.94
ADV_CAP = 0.72
ADV_F = 0.82

# ---- Ridge v6 transition-ledger laws (verbatim lineage) ----
T_ROW_DEST_CAP = 16
T_ROW_TOTAL_CAP = 4096
RELIABILITY_N0 = 8

# ---- Stage A effect/influence laws (verbatim lineage) ----
EFFECT_MIN_BOUTS = 3
EFFECT_MIN_NORM = 0.02
INF_ELIG_DECAY = 0.985
INF_ELIG_AGE = 120
INF_ELIG_FLOOR = 0.08
INF_EVIDENCE_CAP = 0.012
INF_EVIDENCE_GAIN = 0.16
INF_DELTA_THRESH = 0.006
INF_PULSE_THRESH = 0.02

# ---- bout contract (probe-selected in the reference lineage) ----
BOUT_TICKS = 25
PULSE_EXCLUDE_AFTER_RESET = 2

# ---- believed-field law (candidate 2, alignment-audit correction) ----
BELIEF_BETA = 0.5    # per-tick observation-integration rate into B
CHOICE_ETA = 0.30    # backward choice-difference rate into B (B only)

# ---- voxel field (NEW organ; protocol §voxel) ----
VOX_SHAPE = (3, 3, 3)                     # positions in {-1,0,1}^3, seat frame
N_DIRECTIONS = 8                          # cube-corner units
NULL_BASE = 0.25                          # nu_0
KERNEL_POWER = 2                          # kappa = max(0, cos)^power
DIFFUSION_LAMBDA = 0.35
DIFFUSION_ITER = 3

# ---- exploration law (verbatim lineage schedule) ----
EXPLORE_EARLY_TICKS = 900
EXPLORE_EARLY = lambda t: 0.68 * (1 - t / 1100) + 0.18  # noqa: E731
EXPLORE_LATE = lambda err: min(0.46, 0.12 + err * 0.75)  # noqa: E731

TIE_NOISE = 1e-6


# ---- metabolic drive (Stage D; re-grounded homeostasis) ----
METAB_BASAL = 0.001
METAB_K_AWARE = 0.006      # awareness (coherence) burns reserve
METAB_K_CONFIRM = 0.10     # confirmation-under-uncertainty replenishes
METAB_R_TARGET = 0.6
METAB_R_MAX = 1.0
METAB_C_CEIL = 0.95        # awareness ceiling < 1 (never full coherence = death)
