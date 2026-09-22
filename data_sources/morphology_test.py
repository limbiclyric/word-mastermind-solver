"""Detect inflected forms among 5-letter candidates using en_US Hunspell data.
Usage: python3 morphology_test.py
"""
import re, sys
from collections import defaultdict

D = __file__.rsplit('/', 1)[0]

def parse_aff(path):
    groups, cur = {}, None
    for line in open(path, encoding='utf-8'):
        line = line.split('#')[0].rstrip()
        m = re.match(r'SFX (\S+) (\S+) (\d+)', line)
        if m:
            cur = m.group(1); groups[cur] = []; continue
        m = re.match(r'SFX (\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
        if m and m.group(1) in groups:
            strip = m.group(2); add = m.group(3); cond = m.group(4)
            groups[m.group(1)].append((strip, add, cond))
    return groups

def parse_dic(path):
    entries = []
    for i, line in enumerate(open(path, encoding='utf-8')):
        if i == 0: continue
        line = line.strip()
        if not line: continue
        word, _, flags = line.partition('/')
        entries.append((word.lower(), set(flags)))
    return entries

def cond_ok(stem, cond):
    if cond == '.': return True
    if cond.startswith('[') and not cond.startswith('[^'):
        return stem and stem[-1] in cond[1:-1]
    if cond.startswith('[^'):
        return not stem or stem[-1] not in cond[2:-1]
    return stem and stem[-1] == cond

def derived_from(word, entries_by_flag, rules):
    """True if word = some dic stem + one of the given affix rules."""
    for strip, add, cond in rules:
        # word must end in add; stem = word - add + strip
        if not word.endswith(add): continue
        stem = word[:-len(add)] + (strip if strip != '0' else '')
        if not cond_ok(stem, cond): continue
        for w, flags in entries_by_flag.get(stem, ()):
            if any(f in flags for f in rule_flags):
                return True
    return False

aff = parse_aff(f'{D}/en_US.aff')
entries = parse_dic(f'{D}/en_US.dic')
dic = defaultdict(list)
for w, flags in entries:
    if w.isalpha(): dic[w].append(flags)

def inflected(word, group):
    rules = aff[group]
    for strip, add, cond in rules:
        if not word.endswith(add): continue
        stem = word[:-len(add)] + (strip if strip != '0' else '')
        if not cond_ok(stem, cond): continue
        if any(group in flags for flags in dic.get(stem, ())):
            return True
    return False

if __name__ == '__main__':
    answers = {w.strip() for w in open(f'{D}/wordle_answers_actual.txt')}
    cands = [w.strip() for w in open(f'{D}/../possible_answers.txt')]
    groups = {'plural': 'S', 'past': 'D', 'gerund': 'G', 'ly-adverb': 'Y', 'superlative': 'T'}
    keep = {}
    for name, g in groups.items():
        bad = sorted(w for w in answers if inflected(w, g))
        nc = sum(inflected(w, g) for w in cands)
        print(f"{name}: {nc} candidates flagged, actual-answer violations: {bad}")
        keep[name] = not bad
    if '--apply' in sys.argv:
        new = [w for w in cands if not any(inflected(w, g) and keep[n] for n, g in groups.items())]
        open(f'{D}/../possible_answers.txt', 'w').write('\n'.join(new) + '\n')
        print(f"applied -> {len(new)} words")
