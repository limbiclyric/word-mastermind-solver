"""Build ../possible_answers.txt — the Wordle bot answer list.

Pipeline:
  1. Load every answer Wordle has actually published
     (wordle_answers_actual.txt, kept up to date from the wordsolverx archive).
  2. Find the rarest of the answers selected under NYT ownership
     (puzzle #241 onward — the first answer the NYT edited) by wordfreq
     Zipf frequency.
  3. Keep every 5-letter word in wordfreq's full English word list whose
     Zipf frequency is at least that threshold. Rationale: under NYT
     ownership no answer rarer than this has been used, so rarer words are
     excluded.
  4. Drop plural forms (Hunspell SFX S rule) — no published answer has ever
     been a plain plural. Other suffixes (past tense, gerunds, -ly adverbs)
     have appeared as answers and are kept.
  5. Drop proper nouns: a word that appears in the Hunspell en_US dictionary
     only in capitalized form (Paris, Jason, Texas) is a name or place — no
     published answer has ever been one. Words with a lowercase entry
     (china, frank, dutch) are common nouns and stay. Words absent from the
     dictionary are left alone (the frequency filter already judged them).
  6. Verify every actual answer survived; write the result.

Usage: python3 data_sources/build_possible_answers.py
"""
import sys
from wordfreq import zipf_frequency, iter_wordlist

from morphology_test import inflected

D = __file__.rsplit('/', 1)[0]
OUT = f'{D}/../possible_answers.txt'
NYT_START_PUZZLE = 241


def _proper_nouns():
    """Words that en_US.dic lists only in capitalized form: proper nouns.
    Words with a lowercase entry double as common nouns and are not flagged."""
    forms = {}
    for line in open(f'{D}/en_US.dic', errors='ignore'):
        line = line.strip()
        if not line or line[0].isdigit():
            continue
        w = line.split('/')[0]
        forms.setdefault(w.lower(), set()).add(w[0].islower())
    return {w for w, caps in forms.items() if caps == {False}}


def build():
    answers = [w.strip() for w in open(f'{D}/wordle_answers_actual.txt')]
    # Puzzle #241 (Feb 16, 2022) was the first answer the NYT edited; from
    # there on the answers are NYT-curated. Wardle-era answers set a looser
    # bar ("wooer", zipf 1.18) than the NYT will ever use again.
    nyt = answers[NYT_START_PUZZLE - 1:]
    thresh = min(zipf_frequency(w, 'en') for w in nyt)

    cands = [w for w in iter_wordlist('en')
             if len(w) == 5 and w.isascii() and w.isalpha()
             and zipf_frequency(w, 'en') >= thresh]
    cands.sort()

    final = [w for w in cands if not inflected(w, 'S')]

    proper = _proper_nouns()
    n_prop = sum(w in proper for w in final)
    final = [w for w in final if w not in proper]

    missing = set(nyt) - set(final)
    if missing:
        sys.exit(f'ERROR: actual answers lost by pipeline: {sorted(missing)}')

    with open(OUT, 'w') as f:
        f.write('\n'.join(final) + '\n')
    print(f'threshold zipf {thresh}; {len(cands)} frequency-passing, '
          f'{len(cands) - len(final) - n_prop} plurals removed, '
          f'{n_prop} proper nouns removed, {len(final)} -> {OUT}')


if __name__ == '__main__':
    build()
