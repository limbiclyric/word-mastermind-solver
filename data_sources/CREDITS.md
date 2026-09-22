# Word List Sources and Licenses

## wordle_answers_actual.txt
Every published Wordle answer from June 19, 2021 through August 26, 2026.
Scraped from: wordsolverx.com/wordle-answer-archive

## nyt_valid_guesses.txt
Every word (14,855 as of September 2026) the NYT Wordle site accepts as a
guess. Taken directly from the game's JavaScript bundle: the page at
https://www.nytimes.com/games/wordle/index.html loads hashed chunk files
under https://www.nytimes.com/games-assets/v2/ and one of them contains the
list as a long array of quoted 5-letter words. 

Regenerate with `python3 data_sources/extract_nyt_guess_list.py`, which
re-fetches the chunks and rewrites the file. The chunk hashes change when
NYT republishes the game; if the script finds no list, re-scan the chunk
URLs in the index page and update the script (see the docstring).

Legal note: a flat list of English words is treated "fact", not as creative
work (see `Feist v. Rural`). The same list is mirrored widely on
GitHub/npm. Still, this file exists to describe the site, not to replace
the game. 

Note: this project is not affiliated with The New York Times.

## Frequency source: wordfreq
wordfreq combines seven corpora (Wikipedia, subtitles, news, books, web,
Twitter, and Reddit), drops each word's highest and lowest estimate, and averages
the rest. The average of many corpora is hard to skew, which matters here.

Source: https://github.com/rspeer/wordfreq
License: Apache 2.0 (code); data files CC BY-SA 4.0 — derived lists must credit
wordfreq and share alike. Keep this notice if you redistribute.

## possible_answers.txt
Rebuilt with `python3 data_sources/build_possible_answers.py`.
Built in two steps:

1. Frequency filter: take the rarest word selected during NYT ownership
   (puzzle #241 onward — the first answer the NYT edited), ranked by wordfreq
   Zipf frequency (rarest: "elate", zipf 1.33), then keep every 5-letter word
   in wordfreq's full English word list with zipf >= that threshold. Yields
   25,146 candidates and contains all NYT-era answers.
2. Morphology filter: remove plural forms, detected with the Hunspell
   en_US dictionary (see below). Removes 1,407 plurals; final list 23,739
   words, all NYT-era actual answers still included. (One Wardle-era answer,
   "wooer", falls below the NYT-era bar and is excluded: the NYT has not
   picked a word that rare since it took over.)
3. Proper-noun filter: remove words the Hunspell dictionary lists only in
   capitalized form (Paris, Jason, Texas) -- names and places. No published
   answer has ever been a proper noun. Words with a lowercase entry (china,
   frank, dutch) double as common nouns and stay. Removes 1,258 words,
   including 267 that also pass the NYT accepted-guess gate; no actual
   answer is affected. Words absent from the dictionary are left alone.

The no-plurals rule is empirical: of the 1,872 answers published to date,
none are plural. Other suffix rules were tested and rejected — past-tense
("fried"), gerund ("doing", "thing"), and -ly adverbs ("badly", "early")
have all appeared as answers, so they stay. Run morphology_test.py to
re-verify the rules against the current answer set.

Caveat: irregular plurals (e.g. "geese") are separate dictionary entries and
are not removed. None have been used as an answer.

## Morphology data: en_US.dic / en_US.aff
Hunspell US English dictionary used by morphology_test.py to detect plural
forms via the SFX S affix rules.

Source: https://github.com/libreoffice/dictionaries (en/en_US.dic, en/en_US.aff)
Maintainer: Marco A.G. Pinto (dictionary maintainer)
License: tri-licensed under GNU GPL, GNU LGPL, and MPL (your choice of the
three) — see the repository's README and COPYING files for terms. Credit
LibreOffice/dictionaries if you redistribute these files or lists derived
with them.

## Original Wardle lists
`valid_guesses.txt` is Josh Wardle's original guess list (widely mirrored,
e.g. cfreshman gists). His original answer list was replaced by the rebuilt
`possible_answers.txt` above. Word lists of this kind are generally treated
as facts, not creative works.
