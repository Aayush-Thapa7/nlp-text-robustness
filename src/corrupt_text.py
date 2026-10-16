"""
src/corrupt_text.py
===================
Milestone 4 -- Character-Level Text Corruption Utilities
---------------------------------------------------------
Provides reusable functions that introduce controlled noise into text strings
at the **character level**.  These utilities are the building blocks for the
robustness experiment: we take the clean IMDb test reviews, corrupt them to
varying degrees, and measure how much the model's accuracy drops.

Four corruption strategies are implemented
------------------------------------------
  1. deletion      -- randomly remove characters
  2. insertion     -- insert random lowercase letters at random positions
  3. substitution  -- replace characters with random lowercase letters
  4. transposition -- swap pairs of adjacent characters

What does "corruption rate" mean here?
---------------------------------------
The ``rate`` parameter is a float in [0.0, 1.0] representing the *expected
fraction of characters* that are affected.  For every character in the text
we independently flip a biased coin:

  * With probability ``rate``  --> the corruption operation is applied.
  * With probability 1 - rate --> the character is left unchanged.

This is a Bernoulli trial per character (equivalently, the *total* number of
corrupted characters follows a Binomial distribution).

  rate = 0.00  ->  0%  of characters affected  ->  text is unchanged
  rate = 0.05  ->  5%  of characters affected  ->  very mild noise
  rate = 0.10  -> 10%  of characters affected  ->  moderate noise
  rate = 0.20  -> 20%  of characters affected  ->  heavy noise

Reproducibility
---------------
Every function accepts an optional ``seed`` parameter.  When provided, a
fresh ``random.Random`` instance is seeded with that value.  This means:
  * Two calls with the same (text, strategy, rate, seed) always agree.
  * A call does NOT affect the global random state of the caller.

Edge cases
----------
  * Empty string ("")      --> returned immediately, unchanged.
  * Length-1 string ("x") --> returned unchanged for transposition (no neighbour).
  * rate = 0.0             --> fast path: input returned unchanged.
  * rate = 1.0             --> every character is affected (maximum corruption).

Usage
-----
From the project root (with .venv active):

    from src.corrupt_text import corrupt_text, apply_corruption

    # Single string:
    noisy = corrupt_text("Hello world", strategy="deletion", rate=0.10, seed=42)

    # Batch (list of strings):
    noisy_list = apply_corruption(reviews, strategy="substitution", rate=0.05, seed=0)

Requirements
------------
    Python standard library only (no third-party packages needed).
"""

# ---- Standard-library imports -----------------------------------------------
import random  # Local RNG instances for reproducibility
import string  # alphabet constant

# ---- Module-level constants --------------------------------------------------

# Characters used when generating random replacements / insertions.
# Restricting to lowercase ASCII letters keeps noise plausible and avoids
# accidentally introducing digits or punctuation.
_ALPHABET: str = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'

# Supported strategy names -- exposed so callers can enumerate them.
SUPPORTED_STRATEGIES: tuple[str, ...] = (
    "deletion",
    "insertion",
    "substitution",
    "transposition",
)


# ---- Internal helpers --------------------------------------------------------

def _make_rng(seed: int | None) -> random.Random:
    """
    Create a fresh ``random.Random`` instance seeded with *seed*.

    Using a *local* RNG (rather than the module-level ``random`` functions)
    means corruption calls do not interfere with each other or with the
    caller's random state, and results are reproducible for a given seed.

    Parameters
    ----------
    seed : int or None
        Integer seed for determinism, or None for OS-sourced entropy.
    """
    rng = random.Random()
    rng.seed(seed)
    return rng


def _validate_rate(rate: float) -> None:
    """Raise ValueError if *rate* is outside the valid range [0.0, 1.0]."""
    if not (0.0 <= rate <= 1.0):
        raise ValueError(
            f"Corruption rate must be in [0.0, 1.0], got {rate!r}."
        )


# ---- Individual corruption functions ----------------------------------------

def corrupt_deletion(text: str, rate: float, seed: int | None = None) -> str:
    """
    Randomly delete characters from *text*.

    Each character is independently omitted with probability ``rate``.
    Whitespace is included as a candidate, so at high rates even spaces
    between words may disappear.

    Parameters
    ----------
    text : str
        The original text.
    rate : float
        Probability in [0, 1] that each character is deleted.
    seed : int or None
        Reproducibility seed.

    Returns
    -------
    str
        Text with approximately ``rate * len(text)`` characters removed.
        Returns ``""`` when all characters are deleted or when input is empty.

    Examples
    --------
    >>> corrupt_deletion("hello", rate=0.0, seed=42)
    'hello'
    >>> corrupt_deletion("hello", rate=1.0, seed=42)
    ''
    """
    if not text:
        return text
    _validate_rate(rate)
    if rate == 0.0:
        return text  # Fast path: nothing to delete

    rng = _make_rng(seed)
    # Keep a character only when the random draw is >= rate (NOT deleted)
    return "".join(ch for ch in text if rng.random() >= rate)


def corrupt_insertion(text: str, rate: float, seed: int | None = None) -> str:
    """
    Randomly insert extra lowercase letters into *text*.

    For each character in the original string, with probability ``rate`` we
    insert one random lowercase letter *before* that character.  The original
    character is always retained, so the output is always at least as long as
    the input.

    On average, ``len(text) * rate`` extra characters are inserted.

    Parameters
    ----------
    text : str
        The original text.
    rate : float
        Probability in [0, 1] that a random letter is inserted before each
        original character.
    seed : int or None
        Reproducibility seed.

    Returns
    -------
    str
        Text with extra characters inserted.  Length is between len(text)
        and 2 * len(text).

    Examples
    --------
    >>> corrupt_insertion("hi", rate=0.0, seed=0)
    'hi'
    """
    if not text:
        return text
    _validate_rate(rate)
    if rate == 0.0:
        return text

    rng = _make_rng(seed)
    result: list[str] = []
    for ch in text:
        if rng.random() < rate:
            result.append(rng.choice(_ALPHABET))  # Insert a random letter...
        result.append(ch)                          # ...then keep the original
    return "".join(result)


def corrupt_substitution(text: str, rate: float, seed: int | None = None) -> str:
    """
    Randomly replace characters in *text* with random lowercase letters.

    Each character is replaced with probability ``rate``.  The replacement is
    drawn uniformly from the 26 lowercase ASCII letters.  The output is always
    the same length as the input.

    Parameters
    ----------
    text : str
        The original text.
    rate : float
        Probability in [0, 1] that each character is replaced.
    seed : int or None
        Reproducibility seed.

    Returns
    -------
    str
        Text with approximately ``rate * len(text)`` characters substituted.

    Examples
    --------
    >>> corrupt_substitution("abc", rate=0.0, seed=99)
    'abc'
    """
    if not text:
        return text
    _validate_rate(rate)
    if rate == 0.0:
        return text

    rng = _make_rng(seed)
    result: list[str] = []
    for ch in text:
        if rng.random() < rate:
            result.append(rng.choice(_ALPHABET))  # Replace with random letter
        else:
            result.append(ch)                      # Keep original
    return "".join(result)


def corrupt_transposition(text: str, rate: float, seed: int | None = None) -> str:
    """
    Randomly swap pairs of adjacent characters in *text*.

    We scan the text left to right.  At each position ``i``, with probability
    ``rate`` we swap characters at ``i`` and ``i + 1`` and then skip to
    ``i + 2`` (so the just-swapped character is not involved in another swap
    immediately).  The output is always the same length as the input.

    This simulates the most common real-world keyboard typo: pressing two
    adjacent keys in the wrong order (e.g., "teh" instead of "the").

    Parameters
    ----------
    text : str
        The original text.
    rate : float
        Probability in [0, 1] that each character triggers a swap with its
        right neighbour.
    seed : int or None
        Reproducibility seed.

    Returns
    -------
    str
        Text with some adjacent character pairs swapped.  Same length as input.
        Strings shorter than 2 characters are returned unchanged.

    Examples
    --------
    >>> corrupt_transposition("abcde", rate=0.0, seed=7)
    'abcde'
    """
    if len(text) < 2:
        return text  # Nothing to swap
    _validate_rate(rate)
    if rate == 0.0:
        return text

    rng = _make_rng(seed)
    chars = list(text)  # Convert to list so we can do in-place swaps
    i = 0
    while i < len(chars) - 1:
        if rng.random() < rate:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]  # Swap neighbours
            i += 2  # Skip: the swapped pair cannot participate again immediately
        else:
            i += 1
    return "".join(chars)


# ---- Strategy dispatch table ------------------------------------------------

# Maps strategy name -> implementation function.
# Adding a new strategy in the future is as simple as writing a new function
# and adding one entry here.
_STRATEGY_MAP: dict = {
    "deletion":      corrupt_deletion,
    "insertion":     corrupt_insertion,
    "substitution":  corrupt_substitution,
    "transposition": corrupt_transposition,
}


# ---- Public API: single-string entry point ----------------------------------

def corrupt_text(
    text: str,
    strategy: str,
    rate: float,
    seed: int | None = None,
) -> str:
    """
    Apply one character-level corruption strategy to a single string.

    This is the primary entry point for corrupting individual texts.  It
    validates the ``strategy`` and ``rate``, then dispatches to the correct
    low-level function.

    Parameters
    ----------
    text : str
        The original text.  Empty strings are returned unchanged.
    strategy : str
        Which corruption to apply.  One of:
          * ``"deletion"``      -- randomly remove characters
          * ``"insertion"``     -- randomly insert characters
          * ``"substitution"``  -- randomly replace characters
          * ``"transposition"`` -- randomly swap adjacent characters
    rate : float
        Corruption intensity in [0.0, 1.0].
          * 0.0 --> text is returned unchanged (identity operation).
          * 1.0 --> every character is affected (maximum corruption).
    seed : int or None
        Optional integer for reproducibility.  The same
        (text, strategy, rate, seed) quartet always produces the same output.

    Returns
    -------
    str
        The corrupted string.

    Raises
    ------
    ValueError
        If ``strategy`` is not in SUPPORTED_STRATEGIES, or ``rate`` is
        outside [0.0, 1.0].

    Examples
    --------
    >>> corrupt_text("The movie was great", "deletion", 0.10, seed=42)
    'Te movie was great'
    >>> corrupt_text("", "deletion", 0.5)
    ''
    >>> corrupt_text("hello", "substitution", 0.0, seed=0)
    'hello'
    """
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Supported: {SUPPORTED_STRATEGIES}."
        )
    return _STRATEGY_MAP[strategy](text, rate, seed)


# ---- Public API: batch entry point ------------------------------------------

def apply_corruption(
    texts: list[str],
    strategy: str,
    rate: float,
    seed: int | None = None,
) -> list[str]:
    """
    Apply a corruption strategy to every string in a list.

    Each string is corrupted independently using a per-item seed derived from
    the base ``seed`` plus the item's index.  This guarantees:

      * The same (texts, strategy, rate, seed) always produces the same output.
      * Corrupting ``texts[0]`` is independent of corrupting ``texts[1]``.
      * Inserting or removing items does not alter the corruption of other items
        (because each item's seed is ``seed + index``, not a shared counter).

    At ``rate == 0.0`` this returns a copy of *texts* without any work.

    Parameters
    ----------
    texts : list[str]
        Original strings to corrupt.
    strategy : str
        Corruption strategy name (same values as :func:`corrupt_text`).
    rate : float
        Corruption rate in [0.0, 1.0].
    seed : int or None
        Base random seed.  Per-item seeds are ``seed + index`` when ``seed``
        is not None, or None (OS entropy) when ``seed`` is None.

    Returns
    -------
    list[str]
        New list, same length as *texts*, with each string corrupted.

    Examples
    --------
    >>> apply_corruption(["good", "bad"], "deletion", rate=0.0, seed=0)
    ['good', 'bad']
    """
    # Validate once before the loop -- fail fast
    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Supported: {SUPPORTED_STRATEGIES}."
        )
    _validate_rate(rate)

    if rate == 0.0:
        return list(texts)  # Fast path: no corruption needed

    func = _STRATEGY_MAP[strategy]
    corrupted: list[str] = []
    for idx, text in enumerate(texts):
        # Derive a per-item seed so items are independently reproducible.
        item_seed = (seed + idx) if seed is not None else None
        corrupted.append(func(text, rate, item_seed))
    return corrupted
