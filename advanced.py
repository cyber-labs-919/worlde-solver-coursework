import itertools

import numpy as np

import core
from core import GuessResults, filter_words_fast
from math import log2

POSSIBLE_PATTERNS = list(itertools.product([GuessResults.NOT_PRESENT, GuessResults.PRESENT, GuessResults.CORRECT], repeat=core.WORD_LEN))


def test_calculate_information(word_indexes, word, pattern_matrix):
    pattern__chance_map = {}
    pattern_ind__info_map = {}

    # word_index = core.WORD_TO_INDEX[word]
    # # for pattern_ternary in range(3**5):
    # #     valid_word_indexes = word_indexes[pattern_matrix[word_index, word_indexes] == pattern_ternary]
    #
    # # VECTRORIZE THAT to make it faster
    # pattern_valid_word_matrix = np.zeros((3 ** 5, len(word_indexes)), dtype=np.uint8)
    # # then what???

    for pattern_ternary in range(3**core.WORD_LEN):
        word_index = core.WORD_TO_INDEX[word]
        valid_word_indexes = filter_words_fast(word_indexes, word_index, pattern_ternary, pattern_matrix)
        if len(valid_word_indexes) == 0:
            continue

        chance = len(valid_word_indexes) / len(word_indexes)
        pattern__chance_map[pattern_ternary] = chance
        info = log2(len(word_indexes) / len(valid_word_indexes))
        pattern_ind__info_map[pattern_ternary] = info

    return pattern__chance_map, pattern_ind__info_map

"""
entropy is the expected value of a guess - the expected information it will give us (the amount of bits)
Basically - the expected amount of information it will give us
This is calculated by multiplying each pattern's chance by the bits of information we get from it and adding it up
So for example, if there's a 1% chance that it will give us 10 bits of information, and a 99% chance that it will give us 2 bits of information - the entropy is 10*0.01+2*0.99=2.08, since while there is a chance that it will give us 10 whole bits - we will still most likely get only 2 bits.
"""
def calc_entropy_slow(p_chance_map, p_info_map):
    entropy = 0
    
    for i in range(len(POSSIBLE_PATTERNS)):
        if i not in p_info_map or i not in p_chance_map:
            continue
        entropy += p_info_map[i] * p_chance_map[i]

    return entropy

def calc_entropy_fast(p_chance_map, p_info_map):
    entropy = 0

    for i in range(3**core.WORD_LEN):
        if i not in p_info_map or i not in p_chance_map:
            continue
        entropy += p_info_map[i] * p_chance_map[i]

    # vectorize this (works, but idk if it's faster)
    # p_info_vals = []
    # p_chance_vals = []
    # for k in p_info_map:
    #     p_info_vals.append(p_info_map[k])
    #     p_chance_vals.append(p_chance_map[k])
    # entropy = np.sum(np.array(p_info_vals) * np.array(p_chance_vals))

    return entropy