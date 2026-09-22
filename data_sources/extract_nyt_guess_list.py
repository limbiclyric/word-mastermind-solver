#!/usr/bin/env python3
"""Regenerate ../nyt_valid_guesses.txt — every word the NYT Wordle site
accepts as a guess (14,855 words as of Sept 2026).

The list is shipped in the game's JavaScript bundle. The page at
https://www.nytimes.com/games/wordle/index.html links hashed chunk files
under https://www.nytimes.com/games-assets/v2/; the list sits in one of
the larger chunks as a long run of quoted 5-letter words.

Usage: python3 extract_nyt_guess_list.py   (writes ../nyt_valid_guesses.txt)

The chunk hash changes whenever NYT republishes the game. If this script
finds no list, re-fetch the index page and find the chunk containing the
word run (look for the one whose fetched text has a very long
'"aahed","aalii",...' style array), then update CHUNKS below.
"""

import re
import urllib.request
from pathlib import Path

INDEX_URL = "https://www.nytimes.com/games/wordle/index.html"
# Chunks large enough to hold the ~90KB word array (checked Sept 2026).
CHUNKS = [
    "2807.94a240a5ccad30b589d2.js",
    "8326.4f0481e7e3e361ae55ce.js",
    "7035.55caeb3faaa3330461c3.js",
]
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()


def extract(js: str) -> list:
    """Return the longest run of quoted 5-letter words in the chunk."""
    runs = [re.findall(r'"([a-z]{5})"', r) for r in re.findall(r'"(?:[a-z]{5}"[,"\s]+){100,}[a-z]{5}"', js)]
    return max(runs, key=len, default=[])


def main():
    words = []
    for chunk in CHUNKS:
        url = f"https://www.nytimes.com/games-assets/v2/{chunk}"
        words = extract(fetch(url))
        if words:
            print(f"found {len(words)} words in {chunk}")
            break
    if not words:
        raise SystemExit("No word list found — the game bundle likely moved; see docstring.")
    out = Path(__file__).resolve().parent.parent / "nyt_valid_guesses.txt"
    out.write_text("\n".join(words) + "\n")
    print(f"wrote {out} ({len(words)} words)")


if __name__ == "__main__":
    main()
