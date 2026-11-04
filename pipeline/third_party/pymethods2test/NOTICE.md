# pyMethods2Test — vendored

Abdelmadjid, I. & Dyer, R. *pyMethods2Test: A Dataset of Python Tests Mapped to Focal
Methods.* MSR 2025. arXiv:2502.05143.

Source: Zenodo record `14264519` (`https://zenodo.org/api/records/14264519/files/<name>/content`).
License: Apache 2.0 (`LICENSE`). Fetched 2026-09-15.

**The files are unmodified.** `pipeline/pym2t_focal.py` runs the two scripts as CLIs and imports
their functions; it never patches them. SHA-256 at vendoring:

```
208962122c081481a609b55a1e7ecfc0cf87905456e771f7cde9a60602e3372a  find_focals.py
1f83e36208238d407686076e73ae5509a80bf939470e805cd922ce18795118fa  get_context.py
a6f358de5000d1249b5285030a5a1eaa4a22e4584eb7c30d020e4bbefc294706  test_identification.py
e454dfb63f84599a0828da73c7829eba1ea32c7c34250e9e94fd680e9973160d  requirements.txt
```

**Deviation from `requirements.txt`:** the pinned `python-Levenshtein~=0.26.1` has no cp314 wheel
and fails to build. We install `fuzzywuzzy~=0.18.0` with the modern `Levenshtein` package, which
is what `fuzzywuzzy/StringMatcher.py` imports. `pym2t_focal.py` refuses to run if `fuzz` has fallen
back to `difflib`, since that changes every ratio against the `> 50` cutoff.
