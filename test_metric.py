"""Verify the --average / --rms metric chain against hand-computed values.

Constructs a 6-answer state and a "perfect splitter" guess: every answer lands
in its own pattern bucket, so each branch is a singleton whose optimal play is
known by hand (guess #1 is the splitter; a miss costs one more guess, a hit
ends the game). The module's root score must equal the hand value exactly.
"""
import math
import numpy as np
import optimal_solver as M

M._build_pattern_table()
try:
    R = np.sort(np.random.default_rng(7).choice(M.NA, 6, replace=False)).astype(np.int32)
    Rset = set(R.tolist())

    def find_splitter(in_R):
        for g in (R.tolist() if in_R else range(M.NG)):
            if not in_R and g in Rset:
                continue
            if len(set(M.PATTERN_TABLE[g, R].tolist())) == 6:
                return int(g)
        raise AssertionError("no perfect splitter found")

    for in_R in (True, False):
        g = find_splitter(in_R)
        lens = np.where(M.PATTERN_TABLE[g, R] == 242, 1.0, 2.0)  # hand-known game lengths
        M._SQ = 1
        s_avg, _ = M._expectimim_worker(g, R, [], None)
        M._SQ = 2
        s_rms, _ = M._expectimim_worker(g, R, [], None)
        h_avg, h_rms = lens.mean(), math.sqrt((lens ** 2).mean())
        assert math.isclose(s_avg, h_avg, rel_tol=1e-9), (g, s_avg, h_avg)
        assert math.isclose(s_rms, h_rms, rel_tol=1e-9), (g, s_rms, h_rms)
        assert s_rms >= s_avg  # RMS >= mean, always
        print(f"splitter {M.GUESSES[g]!r} (in answers: {in_R}): "
              f"module avg={s_avg:.6f} rms={s_rms:.6f} | hand avg={h_avg:.6f} rms={h_rms:.6f}")

    # Min node must equal the min over its own ranked candidates' chance values.
    M._SQ = 1
    v = M._calc_min_score(R, [], {})
    cands = M._rank_legal_guesses(R, [])
    v2 = min(M._chance_node_score(gi, R, [], float('inf'), {}) for gi in cands)
    assert math.isclose(v, v2, rel_tol=1e-9), (v, v2)
    print(f"min node over {len(cands)} candidates: {v:.6f} == min of chance values {v2:.6f}")

    # Probe: two answers left. True optimum is 1.5 (play one of them:
    # 50% win in 1, 50% win in 2). See what the tree's top-k ranking values it at.
    R2 = np.array(sorted(R[:2].tolist()), dtype=np.int32)
    print(f"2-candidate state: tree value {M._calc_min_score(R2, [], {})} (true optimum 1.5)")
finally:
    M._release_shm()
print("all assertions passed")
