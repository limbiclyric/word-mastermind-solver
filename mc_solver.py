# mc_solver.py
# Monte-Carlo selection of the next word-mastermind guess.


#############
# Constants #
#############

# For the first two, values more than all possible valid words will get reduced to the number of possible valid words.

POTENTIAL_TRUTHS_TO_TEST = 3000      # subsample of still-possible solutions
NEXT_GUESS_TRIALS        = 3000      # candidate guesses to evaluate
TESTS_PER_TRIAL          = 2000      # random completions per (truth, candidate)
WORDS_FILE_PATH          = "nyt_valid_guesses.txt"

###########
# Logging #
###########
import logging
logging.basicConfig(
    filename="wordle_error.log",
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s  %(process)d  %(threadName)s  %(levelname)s  %(message)s",
    force=True
)
log = logging.getLogger(__name__)


#######################
# Word Bank Utilities #
#######################
import numpy as np
from pathlib import Path
import numba as nb
from numba import njit, prange

def load_bank(path=WORDS_FILE_PATH):
    words = [w.strip().lower() for w in Path(path).read_text().splitlines()
             if len(w.strip()) == 5]

    # Convert the concatenated ASCII bytes to an (n_words, 5) uint8 array
    bank_arr = (np.frombuffer(''.join(words).encode('ascii'), dtype=np.uint8)
                  .reshape(-1, 5) - 97)
    return words, bank_arr


###############################
# Monte-Carlo Simulation Core #
###############################

@njit
def _simulate_one(
        truth_idx: int,
        first_idx: int,
        hist_g: np.ndarray,
        hist_c: np.ndarray
        ) -> int:
    """
    Simulate a one game until the true word is found,
    starting with first_idx and respecting past feedback (hist_).
    Returns the number of guesses required.
    """

    if truth_idx == first_idx:
        return 1

    remaining = np.ones(BANK_ARR.shape[0], np.bool_)
    # apply earlier feedback
    for k in range(hist_g.size):
        # Keep only words whose pre-computed feedback against past guess hist_g[k] equals stored code hist_c[k].
        remaining &= (PATTERN_TABLE[hist_g[k]] == hist_c[k])

    pat = PATTERN_TABLE[first_idx, truth_idx]
    if pat == 242: # 242_10 equals 222222_3, that is, the pattern 242 is all green, a correct guess.
        return 1
    remaining &= (PATTERN_TABLE[first_idx] == pat)

    guesses = 1
    while True:
        cand = np.where(remaining)[0]
        if cand.size == 0:
            raise RuntimeError(first_idx, truth_idx)
        pick = cand[np.random.randint(cand.size)]
        guesses += 1
        pat = PATTERN_TABLE[pick, truth_idx]
        if pat == 242:
            return guesses
        remaining &= (PATTERN_TABLE[pick] == pat)

@njit(parallel=True)
def evaluate_candidate_nb(
        first_idx: int,
        truths: np.ndarray,          # int32
        hist_g: np.ndarray,          # int32
        hist_c: np.ndarray,          # uint32
        trials: int
        ) -> float:
        
    m = truths.size
    acc = 0.0
    for i in prange(m):                  # multithreaded
        t = truths[i]
        inner = 0.0
        for _ in range(trials):
            inner += _simulate_one(t, first_idx, hist_g, hist_c)
        acc += inner / trials
    return acc / m # Average length of games.


#############
# Utilities #
#############


# Represent feedback as a base-3 integer.
@nb.njit(cache=True, fastmath=True)
def pattern_code(guess, truth):
    code = 0
    pow3 = 1
    # first pass: greens and letter counts
    taken = np.zeros(26, np.int8)
    for i in range(5):
        g = guess[i]
        if g == truth[i]:
            code += 2 * pow3
        else:
            taken[truth[i]] += 1
        pow3 *= 3
    # second pass: yellows
    pow3 = 1
    for i in range(5):
        g = guess[i]
        if (code // pow3) % 3 == 0 and taken[g]:
            code += 1 * pow3
            taken[g] -= 1
        pow3 *= 3
    return code

# Return matrix M where M[i, j] == pattern_code(bank[i], bank[j])
# NOTE: this builds a full n^2 matrix, where n is the number of possible words.
# This is choosen over a more complex memoization scheme as 2.3K words produces a manageable table, 
# and even the full offical list being around 12K words is manageable.
@nb.njit(parallel=True, cache=True)
def build_pattern_matrix(bank: np.ndarray) -> np.ndarray:    
    n = bank.shape[0]
    mat = np.empty((n, n), np.uint8)
    for i in nb.prange(n):
        for j in range(n):
            mat[i, j] = pattern_code(bank[i], bank[j])
    return mat


#######################
# Monte-Carlo Drivers #
#######################
import sys

WORDS, BANK_ARR = load_bank()
INDEX = {w: i for i, w in enumerate(WORDS)}

PATTERN_TABLE = build_pattern_matrix(BANK_ARR)
BANK_ARR.setflags(write=False)
PATTERN_TABLE.setflags(write=False)

def history_to_arrays(history):
    n = len(history)
    g = np.empty(n, np.int32)
    c = np.empty(n, np.uint32)
    for k, (w, patt) in enumerate(history):
        g[k] = INDEX[w]
        c[k] = sum(p * 3 ** i for i, p in enumerate(patt))
    return g, c


# This is the "public API". What a user interacts with.
def select_next_guess(history=None):
    if history is None:
        history = []
    hist_g, hist_c = history_to_arrays(history)

    still_valid = np.array([
        t for t in range(len(WORDS))
        if all(PATTERN_TABLE[g, t] == c for g, c in zip(hist_g, hist_c))
    ], dtype=np.int32)

    truths = np.random.choice(
        still_valid,
        size=min(POTENTIAL_TRUTHS_TO_TEST, still_valid.size),
        replace=False).astype(np.int32)

    cand = np.random.choice(
        still_valid,
        size=min(NEXT_GUESS_TRIALS, still_valid.size),
        replace=False).astype(np.int32)

    scores = np.empty(cand.size, np.float32)

    try:
        for k, c_idx in enumerate(cand):
            scores[k] = evaluate_candidate_nb(
                c_idx, truths, hist_g, hist_c, TESTS_PER_TRIAL)
    except RuntimeError as err:
        first_idx, truth_idx = err.args
        log.error(
            "No candidates left | first=%s | truth=%s | history=%s",
            WORDS[first_idx],
            WORDS[truth_idx],
            history
        )
        sys.exit(1)
        
    return WORDS[int(cand[scores.argmin()])]


##############################################################
# Main - where we input the history and as for the next word #
##############################################################
# With the following hyper parameters and no history:
#POTENTIAL_TRUTHS_TO_TEST = 3000
#NEXT_GUESS_TRIALS        = 3000
#TESTS_PER_TRIAL          = 2000
# The output is "slate" so we use this as the first guess.

# In the worst case of all greys, the same hyperparameters suggests a second guess of "frond"


import os
if __name__ == "__main__":
    nb.set_num_threads(os.cpu_count() or 1)

    # Example history from playing horsle
    EXAMPLE_HISTORY = [
        ("slate", (1, 0, 0, 0, 2)),
        ("nurse", (0, 0, 2, 2, 2))
    ]
    
    HISTORY = [("frame", (0, 1, 1, 0, 1)), ("paler", (0, 2, 0, 2, 2)), ("wager", (2, 2, 0, 2, 2)), ("waver", (2, 2, 0, 2, 2))]
    
    print(select_next_guess(HISTORY))


