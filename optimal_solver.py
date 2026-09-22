#!/usr/bin/env python3
"""Optimal word-Mastermind solver (default wordlist: NYT Wordle). Scoring heuristic via CLI flag:
  --average  minimize expected number of guesses (expectiminimax)
  --rms      minimize root-mean-square of guesses (risk-averse)
  --cynic    minimize worst-case guesses (minimax with alpha-beta)
No flag: fast/greedy (min expected partition size).
"""
from __future__ import annotations
import atexit, gc, os, time, math, argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Set, Dict
import numba as nb
import numpy as np
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from multiprocessing import shared_memory as _shm_mod

MAX_GUESSES = 6

# ───────────────────────── word lists ─────────────────────────

def _load(p: str) -> List[str]:
    return [w.strip().lower() for w in Path(p).read_text().splitlines() if len(w.strip()) == 5]

ANSWERS = _load("possible_answers.txt")
# Only words the NYT site accepts can be played: gate the answer bank on the
# live accepted-guess list (every real answer is in it, so only words the
# site would refuse are dropped), and take the extras from the same list.
_nyt = set(_load("nyt_valid_guesses.txt"))
ANSWERS = [w for w in ANSWERS if w in _nyt]
_extra_guesses = _load("nyt_valid_guesses.txt")

# GUESSES = answers first (indices stay stable/contiguous), then the rest.
_seen = set()
GUESSES: List[str] = []
for w in ANSWERS + _extra_guesses:
    if w not in _seen:
        _seen.add(w)
        GUESSES.append(w)

NA, NG = len(ANSWERS), len(GUESSES)
G_INDEX = {w: i for i, w in enumerate(GUESSES)}
ANSWER_AS_GIDX = np.fromiter((G_INDEX[w] for w in ANSWERS), dtype=np.int32, count=NA)

# Per-word letter data for the vectorized hard-mode legal-guess filter.
_POW = (1 << np.arange(26, dtype=np.uint32))
GUESS_CODES = (np.frombuffer("".join(GUESSES).encode(), dtype=np.uint8).reshape(-1, 5) - 97).astype(np.int16)
ANSWER_CODES = (np.frombuffer("".join(ANSWERS).encode(), dtype=np.uint8).reshape(-1, 5) - 97).astype(np.int16)
GUESS_MASKS = np.zeros(NG, dtype=np.uint32)
ANSWER_MASKS = np.zeros(NA, dtype=np.uint32)
for _pos in range(5):
    GUESS_MASKS |= _POW[GUESS_CODES[:, _pos]]
    ANSWER_MASKS |= _POW[ANSWER_CODES[:, _pos]]

def _legal_guess_mask(greens, must_include, forbidden_pos, remaining) -> np.ndarray:
    """Boolean mask over all GUESSES: legal under hard mode (greens in place,
    found letters used, no letter in a ruled-out spot, every letter present
    in some remaining answer). Every remaining answer is itself legal."""
    ok = np.ones(NG, dtype=bool)
    if remaining.size:
        allowed = np.bitwise_or.reduce(ANSWER_MASKS[remaining])
        ok &= (GUESS_MASKS & ~allowed) == 0
    m = 0
    for ch in must_include: m |= 1 << (ord(ch) - 97)
    if m: ok &= (GUESS_MASKS & np.uint32(m)) == np.uint32(m)
    for pos, g in enumerate(greens):
        if g: ok &= GUESS_CODES[:, pos] == ord(g) - 97
    for ch, bad in forbidden_pos.items():
        for pos in bad:
            ok &= GUESS_CODES[:, pos] != ord(ch) - 97
    return ok

# ─── Global data (read-only for workers, no locking needed) ───

PATTERN_TABLE: np.ndarray
_SQ = 1  # score power: 1 for --average, 2 for --rms

def _init_globals(shm_name, ng, na, sq):
    """Attach to the shared-memory pattern table (one physical copy for all
    workers) instead of pickling an 800MB array into every process."""
    global PATTERN_TABLE, _SHM, _SQ
    _SQ = sq
    _SHM = _shm_mod.SharedMemory(name=shm_name)
    PATTERN_TABLE = np.frombuffer(_SHM.buf, dtype=np.uint8).reshape(ng, na)
    # Release deterministically before interpreter teardown, where __del__
    # ordering is undefined and may otherwise fire while the array still
    # holds a buffer export on the mmap.
    atexit.register(_worker_shm_cleanup)
    nb.set_num_threads(1)  # pool workers each own a core; inner threads would oversubscribe

def _worker_shm_cleanup():
    global PATTERN_TABLE, _SHM
    PATTERN_TABLE = None
    if _SHM is not None:
        _SHM.close()
        _SHM = None

# ─────────────────── pattern pre-computation ──────────────────

@nb.njit(cache=True, fastmath=True)
def _pattern_code(g: np.ndarray, t: np.ndarray) -> int:
    code, pow3 = 0, 1
    taken = np.zeros(26, np.int8)
    for i in range(5):
        gi, ti = g[i], t[i]
        if gi == ti:
            code += 2 * pow3
        else:
            taken[ti] += 1
        pow3 *= 3
    pow3 = 1
    for i in range(5):
        gi, ti = g[i], t[i]
        if gi != ti and taken[gi] and ((code // pow3) % 3) == 0:
            code += pow3
            taken[gi] -= 1
        pow3 *= 3
    return code

@nb.njit(parallel=True, cache=True)
def _build(tbl_g: np.ndarray, tbl_a: np.ndarray) -> np.ndarray:
    ng, na = tbl_g.shape[0], tbl_a.shape[0]
    out = np.empty((ng, na), np.uint8)
    for i in nb.prange(ng):
        for j in range(na):
            out[i, j] = _pattern_code(tbl_g[i], tbl_a[j])
    return out

# ───────────────────── expectation pass ──────────────────────

nb.set_num_threads(min(os.cpu_count() or 1, 32))

def _expect_all(cand_set: np.ndarray) -> np.ndarray:
    return _expect_rows(PATTERN_TABLE[:, cand_set])

@nb.njit(parallel=True, fastmath=True, cache=True)
def _expect_rows(subtable: np.ndarray) -> np.ndarray:
    """Like _expect_all_cpu_vec, but over a pre-gathered subtable whose rows
    are only the guesses worth scoring."""
    ng, size = subtable.shape
    out = np.empty(ng, np.float32)
    for gi in nb.prange(ng):
        counts = np.bincount(subtable[gi], minlength=243); s = 0.0
        for c in counts: s += c * c
        out[gi] = s / size
    return out

@nb.njit(fastmath=True)
def _row_and_hist(guess_idx: int, cand_set: np.ndarray, row_buf: np.ndarray, hist_buf: np.ndarray, tbl: np.ndarray):
    n = cand_set.size
    row_buf[:n] = tbl[guess_idx][cand_set]
    hist_buf[:] = np.bincount(row_buf[:n].astype(np.int32), minlength=243)

def _adaptive_k(sz: int) -> int:
    if sz <= 25: return sz
    return min(200, max(4, int(np.sqrt(sz) * 3)))

# ─────────────────── hard-mode helpers ──────────────────────

def _decode_pattern(code: int) -> Tuple[int, ...]:
    out = [0] * 5
    for i in range(5):
        out[i] = code % 3
        code //= 3
    return tuple(out)

def _derive_constraints(history: Sequence[Tuple[str, Sequence[int]]]):
    greens = [''] * 5
    must_include = set()
    forbidden_pos = {}
    for guess_word, pattern in history:
        for i, p in enumerate(pattern):
            letter = guess_word[i]
            if p == 2:
                greens[i] = letter
                must_include.add(letter)
            elif p == 1:
                must_include.add(letter)
                forbidden_pos.setdefault(letter, set()).add(i)
    return greens, must_include, forbidden_pos

def _last_guess_won(history: Sequence[Tuple[str, Sequence[int]]]) -> bool:
    """True if the most recent guess was all green (pattern 242): the game already ended."""
    if not history:
        return False
    return sum(p * 3 ** i for i, p in enumerate(history[-1][1])) == 242

def _state_key(remaining: np.ndarray, history: Sequence[Tuple[str, tuple]]) -> tuple:
    """Cache key. The value of a state depends on the candidate set, the depth,
    and the hard-mode constraints implied by the history, so key on all three."""
    greens, must_inc, forb = _derive_constraints(history)
    ckey = (tuple(greens), frozenset(must_inc),
            tuple(sorted((ch, tuple(sorted(bad))) for ch, bad in forb.items())))
    return (tuple(remaining.tolist()), len(history), ckey)

def _rank_legal_guesses(remaining, history, exp_buf=None):
    """Top-k hard-mode-legal candidate guesses, ranked by expected partition
    size. Filter to legal guesses BEFORE scoring: the scoring kernel is
    O(legal x |remaining|) instead of O(NG x |remaining|)."""
    greens, must_inc, forb = _derive_constraints(history)
    legal_rows = np.flatnonzero(_legal_guess_mask(greens, must_inc, forb, remaining))
    k_child = _adaptive_k(len(remaining))
    if legal_rows.size * 4 > NG:
        # Constraints bite weakly here: scoring all guesses and filtering
        # afterwards is cheaper than the subset gather.
        exp = _expect_all(remaining)
        cand_idx = np.argpartition(exp, k_child)[:k_child] if k_child < NG else np.arange(NG)
        cand_idx = sorted(cand_idx, key=lambda i: exp[i])
        legal_set = set(legal_rows.tolist())
        cand_idx = [int(gi) for gi in cand_idx if gi in legal_set]
        if not cand_idx:
            # No legal word in the global top-k: rank the legal ones directly.
            cand_idx = [int(legal_rows[i]) for i in np.argsort(exp[legal_rows])]
    else:
        exp = _expect_rows(PATTERN_TABLE[legal_rows[:, None], remaining])
        if k_child < legal_rows.size:
            top = np.argpartition(exp, k_child)[:k_child]
            order = top[np.argsort(exp[top])]
        else:
            order = np.argsort(exp)
        cand_idx = [int(legal_rows[i]) for i in order]
    return cand_idx

# ─── Expectiminimax search (average / RMS): one algorithm, scores
#     raised to _SQ; the root takes the _SQ-th root for RMS. ───

def _calc_min_score(remaining: np.ndarray, history: List[Tuple[str, tuple]], memo: Dict) -> float:
    """ MIN NODE: Finds the best guess and returns its score (in _SQ power). """
    state_key = _state_key(remaining, history)
    if state_key in memo:
        return memo[state_key]

    depth = len(history)
    if len(remaining) <= 1:
        # One candidate left: if the last guess did not already win, it still
        # costs one more guess to play it.
        d = depth if _last_guess_won(history) else depth + 1
        return float(d) ** _SQ
    if depth >= MAX_GUESSES: return float(MAX_GUESSES + 1) ** _SQ

    cand_idx = _rank_legal_guesses(remaining, history)

    min_score = float('inf')
    prune_threshold = float('inf')

    for guess_idx in cand_idx:
        score = _chance_node_score(guess_idx, remaining, history, prune_threshold, memo)
        min_score = min(min_score, score)
        prune_threshold = min(prune_threshold, min_score)

    memo[state_key] = min_score
    return min_score

def _chance_node_score(guess_idx: int, remaining: np.ndarray, history: List[Tuple[str, tuple]], prune_threshold: float, memo: Dict) -> float:
    """ CHANCE NODE: For a single guess, calculates the average score (in _SQ
    power) over all possible outcomes, with branch-and-bound pruning. """
    row_buf = np.empty(len(remaining), dtype=np.uint8)
    hist_buf = np.empty(243, dtype=np.int32)
    _row_and_hist(guess_idx, remaining, row_buf, hist_buf, PATTERN_TABLE)

    running_weighted_score = 0.0
    n_explored = 0
    total_answers = len(remaining)
    best_possible_future_score = float(len(history) + 1) ** _SQ  # win on the next guess

    outcomes = sorted([(cnt, pat) for pat, cnt in enumerate(hist_buf) if cnt > 0], reverse=True)

    for count, pat_code in outcomes:
        branch = remaining[row_buf[:total_answers] == pat_code]
        new_history = history + [(GUESSES[guess_idx], _decode_pattern(pat_code))]
        score = _calc_min_score(branch, new_history, memo)
        running_weighted_score += count * score
        n_explored += count

        # Optimistic lower bound on the final average: prune if it can't win.
        n_unexplored = total_answers - n_explored
        lower_bound = (running_weighted_score + n_unexplored * best_possible_future_score) / total_answers
        if lower_bound >= prune_threshold:
            return lower_bound

    return running_weighted_score / total_answers

def _expectimim_worker(root_g, possible, history, prune_threshold=None):
    """ Worker that evaluates a single root guess with the _SQ-power metric.
    RMS workers get _SQ=2 via the pool initializer and take the root sqrt here."""
    global _SQ
    if prune_threshold is None: prune_threshold = float('inf')
    memo = {}
    score = _chance_node_score(root_g, possible, history, prune_threshold, memo)
    return (score if _SQ == 1 else math.sqrt(score)), {root_g}

# ─── Minimax search (cynic / worst case) with alpha-beta ───
# Score is a pair: (worst length in guesses, how many answers hit it).
# Lexicographic order, so alpha-beta still prunes; the count breaks the
# massive ties a bare integer worst length produces.

def _admissible_heuristic(cand_set_size: int) -> int:
    if cand_set_size <= 1: return 0
    return math.ceil(math.log(cand_set_size, 243))

def _minimax_recursive(remaining: np.ndarray, history: List[Tuple[str, tuple]], alpha: Tuple[int, int], beta: Tuple[int, int], memo: Dict, exact: bool = True) -> Tuple[int, int]:
    """ MIN node. Returns (worst length, answers hitting it).
    exact=True: tie counts are exact everywhere (slower; used to verify
    candidates). exact=False: fast fail-hard pruning -- lengths stay exact,
    but counts at the pruning boundary are understated."""
    state_key = _state_key(remaining, history)
    if state_key in memo:
        return memo[state_key]

    depth = len(history)
    if len(remaining) <= 1:
        d = depth if _last_guess_won(history) else depth + 1
        return (d, 1)
    if depth >= MAX_GUESSES: return (MAX_GUESSES + 1, len(remaining))

    # Lower bound only (true value >= beta): never memoize a bound.
    if depth + _admissible_heuristic(len(remaining)) >= beta[0]:
        return beta

    cand_idx = _rank_legal_guesses(remaining, history)

    min_eval = beta
    improved = False
    for guess_idx in cand_idx:
        max_eval = (0, 0)
        if exact:
            # Prune only on strictly longer results: the call window is
            # (best_len+1, 0), so any result at or below the current best
            # length is exact and tie counts survive fail-hard bounds.
            beta_call = min(beta, (min_eval[0] + 1, 0))
        else:
            # Fast mode: prune at the current best outright. A result equal to
            # the window comes back as a bound, so exact ties there lose their
            # counts -- but the length stays correct.
            beta_call = min_eval

        row_buf = np.empty(len(remaining), dtype=np.uint8)
        hist_buf = np.empty(243, dtype=np.int32)
        _row_and_hist(guess_idx, remaining, row_buf, hist_buf, PATTERN_TABLE)
        # Smallest branches first: worst cases hide in singleton outcomes, so
        # max_eval reaches min_eval sooner and the loop breaks earlier.
        outcomes = sorted([(cnt, pat) for pat, cnt in enumerate(hist_buf) if cnt > 0])

        for cnt, pat_code in outcomes:
            branch = remaining[row_buf[:len(remaining)] == pat_code]
            new_history = history + [(GUESSES[guess_idx], _decode_pattern(pat_code))]
            # alpha=(0,0) keeps every result below the window exact, so ties at
            # the worst length can sum their counts safely.
            score = _minimax_recursive(branch, new_history, (0, 0), beta_call, memo, exact)
            if not exact and score >= min_eval:
                # Proven not better than the current best: stop costing it out.
                # Keep max_eval at min_eval -- leaving (0,0) here would poison
                # the min with an impossible score.
                max_eval = min_eval
                break
            if score[0] > max_eval[0]:
                max_eval = score
            elif score[0] == max_eval[0]:
                max_eval = (score[0], max_eval[1] + score[1])
            if max_eval >= min_eval: break

        if max_eval < min_eval:
            min_eval = max_eval
            improved = True

    if not improved:
        # No guess beat the incoming bound: min_eval is still that bound, not
        # an exact value -- return it but never memoize it.
        return min_eval
    memo[state_key] = min_eval
    return min_eval

def _greedy_worst_len(remaining, history) -> Tuple[int, int]:
    """ Simulate greedy play (top-ranked legal guess) against every remaining
    answer and return (worst length in guesses, answers hitting it). Cheap and
    achievable -- roots[0] plays exactly this tree -- so it seeds a real
    pruning threshold for the cynic search."""
    depth = len(history)
    if len(remaining) == 1:
        return ((depth, 1) if _last_guess_won(history) else (depth + 1, 1))
    if depth >= MAX_GUESSES: return (MAX_GUESSES + 1, len(remaining))
    gi = _rank_legal_guesses(remaining, history)[0]
    row_buf = np.empty(len(remaining), dtype=np.uint8)
    hist_buf = np.empty(243, dtype=np.int32)
    _row_and_hist(gi, remaining, row_buf, hist_buf, PATTERN_TABLE)
    cur = (0, 0)
    for pat_code in sorted(set(int(p) for p in row_buf[:len(remaining)])):
        branch = remaining[row_buf[:len(remaining)] == pat_code]
        score = _greedy_worst_len(branch, history + [(GUESSES[gi], _decode_pattern(pat_code))])
        if score[0] > cur[0]:
            cur = score
        elif score[0] == cur[0]:
            cur = (cur[0], cur[1] + score[1])
    return cur

def _cynic_worker(root_g, possible, history, prune_threshold=None):
    """ Worker: evaluates one root guess. With prune_threshold=None it returns
    the exact (worst length, answer count). With a threshold of (L+1, 0) it
    runs the fast length-only search: it returns either the marker (L+1, 0)
    (proven at least L+1 -- discard) or a length-exact value to verify."""
    memo = {}  # Each worker gets its own memoization cache
    initial_beta = (MAX_GUESSES + 2, NA + 1)  # unreachable upper bound
    cur = (0, 0)  # current (worst length, answers at it) seen so far
    row_buf = np.empty(len(possible), dtype=np.uint8)
    hist_buf = np.empty(243, dtype=np.int32)
    _row_and_hist(root_g, possible, row_buf, hist_buf, PATTERN_TABLE)
    # Smallest branches first: the worst cases hide in singleton outcomes, so
    # raising `cur` early tightens the window for all the big branches later.
    outcomes = sorted([(cnt, pat) for pat, cnt in enumerate(hist_buf) if cnt > 0])

    if prune_threshold is None:
        for cnt, pat_code in outcomes:
            branch = possible[row_buf[:len(possible)] == pat_code]
            new_history = history + [(GUESSES[root_g], _decode_pattern(pat_code))]
            # Exact mode: search with the tie-safe window (cur_len+1, 0) so
            # ties stay exact, and re-search a strictly worse branch fully.
            score = _minimax_recursive(branch, new_history, (0, 0), (cur[0] + 1, 0), memo, exact=True)
            if score[0] > cur[0]:
                score = _minimax_recursive(branch, new_history, (0, 0), initial_beta, memo, exact=True)
            if score[0] > cur[0]:
                cur = score
            elif score[0] == cur[0]:
                cur = (cur[0], cur[1] + score[1])
    else:
        for cnt, pat_code in outcomes:
            branch = possible[row_buf[:len(possible)] == pat_code]
            new_history = history + [(GUESSES[root_g], _decode_pattern(pat_code))]
            score = _minimax_recursive(branch, new_history, (0, 0), prune_threshold, memo, exact=False)
            if score >= prune_threshold:
                # Proven at least L+1: cheap marker, discard without verifying.
                return prune_threshold, {root_g}
            if score[0] > cur[0]:
                cur = score
            elif score[0] == cur[0]:
                cur = (cur[0], cur[1] + score[1])

    return cur, {root_g}

# ───────────────────── public API ────────────────────────────

def _release_shm():
    """Properly close and unlink the segment. The numba kernels take the
    table as an argument, so nothing pins it and the mmap closes cleanly."""
    global PATTERN_TABLE, _SHM
    if _SHM is None: return
    PATTERN_TABLE = None
    shm, _SHM = _SHM, None
    gc.collect()
    shm.close()
    shm.unlink()

def _build_pattern_table():
    global PATTERN_TABLE, _SHM
    if 'PATTERN_TABLE' not in globals() or PATTERN_TABLE is None:
        a_arr_ = (np.frombuffer("".join(ANSWERS).encode(), dtype=np.uint8).reshape(-1, 5) - 97)
        g_arr_ = (np.frombuffer("".join(GUESSES).encode(), dtype=np.uint8).reshape(-1, 5) - 97)
        # Build into shared memory so every pool worker reads the same
        # physical pages instead of a pickled private copy.
        _SHM = _shm_mod.SharedMemory(create=True, size=NG * NA)
        PATTERN_TABLE = np.frombuffer(_SHM.buf, dtype=np.uint8).reshape(NG, NA)
        PATTERN_TABLE[:] = _build(g_arr_, a_arr_)

def select_next_guess(history: Sequence[Tuple[str, Sequence[int]]],
                      mode: str = "greedy",
                      ) -> List[str]:
    global _SQ
    _SQ = {"average": 1, "rms": 2}.get(mode, 1)
    _build_pattern_table()

    remaining = np.arange(NA, dtype=np.int32)
    for word, patt in history:
        gi = G_INDEX[word]
        code = sum(p * (3 ** i) for i, p in enumerate(patt))
        remaining = remaining[PATTERN_TABLE[gi, remaining] == code]
        if not remaining.size: raise ValueError("Impossible history")

    if remaining.size <= 2:
        _release_shm()
        return [ANSWERS[int(i)] for i in remaining]

    exp_buf = _expect_all(remaining)
    k_root = _adaptive_k(remaining.size)

    if mode == "greedy":
        best_exp_guess = min(range(NG), key=lambda i: exp_buf[i])
        _release_shm()
        return [GUESSES[int(best_exp_guess)]]

    search_func = _cynic_worker if mode == "cynic" else _expectimim_worker
    # Same candidate ranking the tree searches use: hard-mode-legal guesses,
    # ranked by expected partition size (with its built-in fallback when the
    # legal words rank poorly in the global top-k).
    roots = _rank_legal_guesses(remaining, list(history))

    best_score = float('inf')
    best_set: Set[int] = set()
    if not roots: return ["No valid guess found"]

    ctx = get_context("spawn")
    try:
        with ProcessPoolExecutor(mp_context=ctx, initializer=_init_globals, initargs=(_SHM.name, NG, NA, _SQ)) as pool:
            if mode == "cynic":
                # Two stages. Stage 1 (fast): every root runs the fail-hard
                # length-only search pruned against the greedy seed's length;
                # roots proving longer return a cheap marker. Stage 2 (exact):
                # roots at or under the best length get their tie counts
                # verified exactly -- that cost is intrinsic to ranking ties.
                seed_len, seed_cnt = _greedy_worst_len(remaining, list(history))
                print(f"Greedy seed: worst case takes {seed_len} guesses and {seed_cnt} answer(s) reach it")
                best_len = seed_len
                # Every root can be measured against the greedy tree's value:
                # roots proving worse than (seed_len, seed_cnt+1) return a cheap
                # marker, everything below comes back exact.
                threshold = (seed_len, seed_cnt + 1)
                pending = list(roots)
                wave: object = []
                candidates: object = []
                n_wave = min(os.cpu_count() or 1, len(pending))
                while pending or wave:
                    while pending and len(wave) < n_wave:
                        wave.append(pool.submit(search_func, pending.pop(0), remaining, list(history), threshold))
                    done, wave = wait(wave, return_when=FIRST_COMPLETED)
                    wave = list(wave)
                    for fut in done:
                        s, opts = fut.result()
                        rg = list(opts)[0]
                        if s[0] <= best_len:
                            candidates.append(rg)
                print(f"Stage 1 done: {len(candidates)} candidate(s) at worst length {best_len}")
                # Stage 2: exact counts for the candidates, in parallel.
                best_score: object = (best_len, NA + 1)
                for fut in [pool.submit(search_func, rg, remaining, list(history), None) for rg in candidates]:
                    s, opts = fut.result()
                    if s < best_score:
                        best_score, best_set = s, set(opts)
                    elif s == best_score:
                        best_set.update(opts)
            else:
                # Evaluate the first root serially to get a baseline score, then
                # evaluate the rest in parallel using it as the pruning threshold.
                first_root = roots.pop(0)
                best_score, first_set = search_func(first_root, remaining, list(history), None)
                best_set.update(first_set)
                print(f"Baseline score from '{GUESSES[first_root]}': {best_score}")
                threshold = best_score
                tasks = [pool.submit(search_func, rg, remaining, list(history), threshold) for rg in roots]
                for fut in tasks:
                    s, opts = fut.result()
                    # 's' can be an optimistic (pruned) lower bound, so verify
                    # exactly whenever it looks at least as good as the best.
                    if s <= best_score:
                        rg = list(opts)[0]
                        true_s, _ = search_func(rg, remaining, list(history), None)
                        if true_s < best_score:
                            print(f"  Found better candidate '{GUESSES[rg]}' with score {true_s} (current best: {best_score})")
                            best_score, best_set = true_s, {rg}
                        elif true_s == best_score:
                            best_set.update(opts)
    finally:
        _release_shm()

    if mode == "cynic":
        print(f"Best worst-case: {best_score[0]} guesses, {best_score[1]} answers (ties: {len(best_set)})")
    final_candidates = sorted(list(best_set), key=lambda i: (
        exp_buf[i], -1 if GUESSES[i] in ANSWERS else 0, GUESSES[i]))
    return [GUESSES[i] for i in final_candidates]



# Best Starting Words
# Cynic:    'larns', 'slate'
#           requires more than six guesses (i.e. game over) in 42 cases
# RMS:      'salet' with score 4.055633125262808
# Average:  'salet' and 'teals' tie with average guesses = 3.9591121495327104           

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimal word-Mastermind solver (default wordlist: NYT Wordle). Scoring heuristic flags (default: fast/greedy).")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--average", action="store_true", help="Minimize expected guesses (expectiminimax).")
    g.add_argument("--rms", action="store_true", help="Minimize root-mean-square of guesses (risk-averse).")
    g.add_argument("--cynic", action="store_true", help="Minimize worst-case guesses (minimax).")
    args = parser.parse_args()
    mode = "average" if args.average else "rms" if args.rms else "cynic" if args.cynic else "greedy"

    HISTORY: List[Tuple[str, Sequence[int]]] = []

    t0 = time.time()
    next_guess = select_next_guess(HISTORY, mode=mode)
    print(f"Mode: {mode}")
    print("Best next guess(es):", next_guess)
    print(f"Total Elapsed {time.time() - t0:.2f}s")
