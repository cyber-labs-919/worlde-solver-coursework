import hashlib
from enum import Enum
import os
import itertools as it
import numpy as np

filename = "words.txt"

WORD_LEN = 5

with open(filename, 'r') as f:
    ALL_WORDS = f.read().splitlines()
    WORD_TO_INDEX = {w: i for i, w in enumerate(ALL_WORDS)}

class GuessResults(Enum):
    NOT_PRESENT = 0  # Letter not present
    PRESENT = 1  # Letter present
    CORRECT = 2  # Letter correct

def format_results(results):
    """
    Helper function that converts the information to its string representation.
    """
    return "".join(str(s.value) for s in results)

def process_guess_slow(guess, target):
    """
    Processes the guess and returns the information gained about each letter in a list.
    
    Args:
        guess::[str]  
            The word that was guessed.
        target::[str]  
            The target word, the correct answer.

    Returns:
        [List[GuessResults]]  
            A list of GuessResults (same length as the word), each one representing the guess result for the respective letter.
    """
    
    results = []
    for c_i in range(len(guess)):
        c = guess[c_i]
        if target[c_i] == c:
            results.append(GuessResults.CORRECT)
        elif c in target:
            results.append(GuessResults.PRESENT)
        else:
            results.append(GuessResults.NOT_PRESENT)
            
    return results


def process_guess(guess, target, pattern_matrix: np.ndarray):
    return pattern_matrix[WORD_TO_INDEX[guess], WORD_TO_INDEX[target]]


def is_victory(results):
    """
    Helper function that returns whether or not the guess was correct, according to the information gained from the result.
    """
    return all(x == GuessResults.CORRECT for x in results)


MISS = np.uint8(0)
MISPLACED = np.uint8(1)
EXACT = np.uint8(2)

def words_to_int_arrays(words):
    return np.array([[ord(c)for c in w] for w in words], dtype=np.uint8)

def generate_pattern_matrix(words1, words2):
    """
    A pattern for two words represents the wordle-similarity
    pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
    between 0 and 3^5. Reading this integer in ternary gives the
    associated pattern.
    This function computes the pairwise patterns between two lists
    of words, returning the result as a grid of hash values. Since
    this can be time-consuming, many operations that can be are vectorized
    (perhaps at the expense of easier readibility), and the the result
    is saved to file so that this only needs to be evaluated once, and
    all remaining pattern matching is a lookup.
    """


    # Number of letters/words
    nl = len(words1[0])
    nw1 = len(words1)  # Number of words
    nw2 = len(words2)  # Number of words

    # Convert word lists to integer arrays
    word_arr1, word_arr2 = map(words_to_int_arrays, (words1, words2))

    # equality_grid keeps track of all equalities between all pairs
    # of letters in words. Specifically, equality_grid[a, b, i, j]
    # is true when words[i][a] == words[b][j]

    # basically, this is a matrix of matrices, where each matrix represents indexes of matching letters between the two words
    # so the indexing looks like this
    # equality_grid[word_a_index,word_b_index,letter_index_for_word_a,letter_index_for_word_b]
    equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    for i, j in it.product(range(nl), range(nl)):
        equality_grid[:, :, i, j] = np.equal.outer(word_arr1[:, i], word_arr2[:, j])

    # ["aaaaa", "qooow"], ["qwert", "ddddd", "eeeee"]
    # results

    # full_pattern_matrix[a, b] should represent the 5-color pattern
    # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
    full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

    # Green pass
    # we could put this after yellow pass if it wasn't for the fact that
    #   we need to remove equivalences in the yellow pass for the special rule about repeating matches
    #   (see below)
    # because of that, we need to do the green pass first and simply set the equivalence for the
    #   green letters to false
    for i in range(nl):
        matches = equality_grid[:, :, i, i].flatten()  # matches[a, b] is true when words[a][i] = words[b][i]
        full_pattern_matrix[:, :, i].flat[matches] = EXACT

        for k in range(nl):
            # If it's a match, mark all elements associated with
            # that letter, both from the guess and answer, as covered.
            # That way, it won't trigger the yellow pass.
            equality_grid[:, :, k, i].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Yellow pass
    for i, j in it.product(range(nl), range(nl)):
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = MISPLACED

        # we want to make sure if for example the word is "river" and we guess
        #   "brrrb" - the first two r's are yellow and the third is grey
        # to do that, once a letter matches - we remove it, so that the second time it only
        #   considers it right if there's yet another match
        for k in range(nl):
            equality_grid[:, :, k, j].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False


    # Rather than representing a color pattern as a lists of integers,
    # store it as a single integer, whose ternary representations corresponds
    # to that list of integers.

    # we're basically converting to base 3
    pattern_matrix = np.dot(
        full_pattern_matrix,
        (3**np.arange(nl)).astype(np.uint8)
    )

    return pattern_matrix

def get_pattern_matrix(words):
    h = hashlib.md5(str(words).encode(encoding="utf8")).hexdigest()
    if os.path.exists(f"pattern_matrix_{h}.npy"):
        return np.load(f"pattern_matrix_{h}.npy")

    pattern_matrix = generate_pattern_matrix(words, words)
    np.save(f"pattern_matrix_{h}.npy", pattern_matrix)
    return pattern_matrix



def filter_words_slow(words, guess, result_pattern):
    """
    NOTE: do not use this. This is an outdated, slow version of this function.
    Takes a list of words and returns a new list, filtering out words that don't match the information we've gained from a guess.

    Args:
        words::[List[str]]
            List of the currently viable words that will be filtered according to the guess, and results gained from that guess.
        guess::[str]
            The word that was guessed.
        result_pattern::[List[GuessResults]]
            The information (pattern) gained for that guess.

    Returns:
        [List[str]]
            A copy of the original list that was filtered according to the newly gained information.
    """
    final_words = []

    for word in words:
        exclude = False
        for c_i in range(WORD_LEN):
            c = guess[c_i]
            r = result_pattern[c_i]

            if r == GuessResults.CORRECT:
                if word[c_i] != c:
                    exclude = True
                    break
            elif r == GuessResults.PRESENT:
                if c not in word or word[c_i] == c:
                    exclude = True
                    break
            else:
                if c in word:
                    exclude = True
                    break
        if not exclude:
            final_words.append(word)

    # print(f"filter_words_slow took {time.time() - start} seconds for {len(words)} words, guess {guess} and pattern {result_pattern}")
    return final_words

def filter_words_slow_2(words, guess, result_pattern):
    """
    NOTE: do not use this. This is an outdated, slow version of this function.
    Same as filter_words_slow, but with a different implementation.
    """
    final_words = []

    for considered_target in words:
        pattern_we_would_get = process_guess_slow(guess, considered_target)
        if tuple(pattern_we_would_get) == tuple(result_pattern):
            final_words.append(considered_target)

    # print(f"filter_words_slow took {time.time() - start} seconds for {len(words)} words, guess {guess} and pattern {result_pattern}")
    return final_words

def filter_words_fast(word_indexes, guess_index, result_pattern_ternary: int, pattern_matrix):
    """
    Takes a list of words and returns a new list, filtering out words that don't match the information we've gained from a guess.

    Args:
        word_indexes::[np.ndarray]
            List of indexes for the currently viable words that will be filtered according to the guess, and results gained from that guess.
        guess_index::[int]
            The index of the word that was guessed.
        result_pattern_ternary::[int]
            The information (pattern) gained for that guess, as an int (the base 3 number for the pattern converted to decimal).

    Returns:
        [np.ndarray]
            A copy of the original list that was filtered according to the newly gained information.
    """

    # final_words = []
    # for considered_target_index in word_indexes:
    #     pattern_we_would_get = pattern_matrix[guess_index, considered_target_index]
    #     if pattern_we_would_get == result_pattern_ternary:
    #         final_words.append(considered_target_index)

    # if we vectorize that with numpy...
    # but keep in mind that word_indexes can have a different length
    final_words = word_indexes[pattern_matrix[guess_index, word_indexes] == result_pattern_ternary]

    # then we go from spending around 10 seconds per word... to spending 10 MILLISECONDS per word!

    return final_words
