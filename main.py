import math
import os
from enum import Enum
import pickle
import random
import matplotlib.pyplot as plt

import numpy as np

import core
import advanced
import hashlib

class GameModes(Enum):
    VERSUS = 1
    PLAYER_ONLY = 2
    PC_ONLY = 3

class Game:
    def __init__(self, mode):
        self.mode = mode
        
        self.player_won = False
        self.pc_won = False
    
        self.target_pc = None
        self.target_player = None
        
        self.is_first_turn = False
        
        self.format_hints_function = None
        self.player_hints_possible_letters = None
        self.player_hints_invalid_letters = None
        self.player_hints_existing_letters = None
        self.player_hints_word_mask = None
        self.player_hints_possible_letters = "\nqwertyuiop\n asdfghjkl\n  zxcvbnm\n"
        self.player_hints_existing_letters = ""
        self.player_hints_invalid_letters = ""
        self.player_hints_word_mask = "*****"
        
        # A list of all possible words for the computer
        self.pc_words = [w for w in core.ALL_WORDS]
        self.pattern_matrix = core.get_pattern_matrix(core.ALL_WORDS)

        self.turns = 0

    def format_ui(self):
        return (
        f"Unknown: {''.join(self.player_hints_possible_letters)}\n"
    f"Correct: {''.join(self.player_hints_existing_letters)}\n"
    f"Invalid: {''.join(self.player_hints_invalid_letters)}\n"
    f"Word: {self.player_hints_word_mask}"
        )

    def get_player_word(self):
        while True:
            word = input("Select your word: ")
            if word in core.ALL_WORDS:
                return word
            else:
                print("Invalid word, try again.")
    
    def get_player_guess(self):
        while True:
            guess = input("Your guess: ")
            if guess in core.ALL_WORDS:
                return guess
            else:
                print("Invalid word, try again.")

    def _update_ui(self, guess, results):
        """
        Updates the user UI data according to the information gained from the guess.
        """
        for c_i in range(len(guess)):
            c = guess[c_i]
            r = results[c_i]
            if r == core.GuessResults.NOT_PRESENT:
                n = ""
                for l in self.player_hints_possible_letters:
                    n += "•" if l == c else l
                self.player_hints_possible_letters = n
                if c not in self.player_hints_invalid_letters:
                    self.player_hints_invalid_letters += c
            else:
                if c not in self.player_hints_existing_letters:
                    self.player_hints_existing_letters += c
                n = ""
                for l in self.player_hints_possible_letters:
                    if l == c:
                        if r == core.GuessResults.NOT_PRESENT:
                            n += "•"
                        else:
                            n += l.upper()
                    else:
                        n += l
                self.player_hints_possible_letters = n
                if r == core.GuessResults.CORRECT:
                    p = ""
                    p += self.player_hints_word_mask[:c_i]
                    p += c
                    p += self.player_hints_word_mask[c_i+1:]
                    self.player_hints_word_mask = p

    def player_turn(self):
        print()
        if not self.is_first_turn:
            print(self.format_ui())
            
        self.is_first_turn = False

        print()
        guess = self.get_player_guess()
    
        results = core.process_guess(guess, self.target_pc, self.pattern_matrix)
        print(f"Results: {core.format_results(results)}")

        self._update_ui(guess, results)
        
        print()
        self.player_won = core.is_victory(results)

    def is_over(self):
        if self.mode == GameModes.VERSUS:
            return self.player_won or self.pc_won
        if self.mode == GameModes.PC_ONLY:
            return self.pc_won
        if self.mode == GameModes.PLAYER_ONLY:
            return self.player_won
        
    def ai_turn_random(self):
        print("No algorithm, choosing a random word.")
        guess = random.choice(self.pc_words)
        self.pc_words = [word for word in self.pc_words if word != guess]
        results = core.process_guess(guess, self.target_player, self.pattern_matrix)
        print(f"Guessing: {guess} (confidence: {100/len(self.pc_words):.2f}%)\nResult: {core.format_results(results)}")
        # Naive algorithm - will explain later
        self.pc_words = core.filter_words_slow(self.pc_words, guess, results)
        self.pc_won = core.is_victory(results)
        

    def play(self):
        if self.mode in (GameModes.PC_ONLY, GameModes.VERSUS):
            self.target_player = self.get_player_word()

        if self.mode in (GameModes.PLAYER_ONLY, GameModes.VERSUS):
            # AI choosing a random word for the player to guess
            self.target_pc = random.choice(core.ALL_WORDS)
            print("The AI has selected its word.\n")
        
        # Randomly choosing who starts first, the user or the AI
        if self.mode == GameModes.PC_ONLY:
            self.is_player_turn = False
        elif self.mode == GameModes.PLAYER_ONLY:
            self.is_player_turn = True
        elif self.mode == GameModes.VERSUS:
            self.is_player_turn = random.choice([True,False])
            if self.is_player_turn:
                print("You guess first.")

        while not self.is_over():
            self.turns += 1

            if self.is_player_turn:
                self.player_turn()
                if self.mode in (GameModes.PC_ONLY, GameModes.VERSUS):
                    self.is_player_turn = not self.is_player_turn
                continue

            if self.pc_won:
                print("The PC has already won.")
            elif self.mode in (GameModes.PC_ONLY, GameModes.VERSUS):
                self.ai_turn_random()
            else:
                print()

            if self.mode != GameModes.PC_ONLY:
                self.is_player_turn = not self.is_player_turn

        if self.player_won:
            print(f"Congrats, you've won in {self.turns} turns!")
        else:
            print(f"It looks like I've won in {self.turns}. My word was {self.target_pc}")

def secs_to_str(secs):
    # days, hours, minutes, seconds
    # omits all zeroes
    if secs <= 0:
        return "0 secs"
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)

    strings = []
    for value, unit in ((d, "day"), (h, "hour"), (m, "min"), (s, "sec")):
        if value <= 0:
            continue
        st = f"{value} {unit}"
        if d != 1:
            st += "s"
        strings.append(st)
    return ", ".join(strings)

def run_all_test_inner(target_words, algorithm_test_method, run_n_times=1, show_progress_every_n_secs=2):
    sum_ = 0
    min_turns = math.inf
    max_turns = 0
    start_t = time.time()

    last_progress_report_time = time.time()
    last_progress_report_ticks = 0

    total_ticks = len(target_words) * run_n_times
    total_words_len = len(target_words)

    estimate_avg_time_aggregate = 0
    estimate_avg_checked_times = 0

    turns__count_map = {}


    for i in range(run_n_times):
        for word_i in range(total_words_len):
            word = target_words[word_i]
            turns = algorithm_test_method(word)

            turns__count_map[turns] = turns__count_map.get(turns, 0) + 1

            sum_ += turns
            if turns < min_turns:
                min_turns = turns
            if turns > max_turns:
                max_turns = turns

            if show_progress_every_n_secs and time.time() - last_progress_report_time > show_progress_every_n_secs:
                progress_elapsed_time = time.time() - last_progress_report_time
                cur_tick = i * total_words_len + word_i
                cur_ticks = cur_tick + 1
                progress_elapsed_ticks = cur_ticks - last_progress_report_ticks
                progress_elapsed_secs_per_tick = progress_elapsed_time / progress_elapsed_ticks

                estimate_avg_time_aggregate += progress_elapsed_secs_per_tick
                estimate_avg_checked_times += 1

                estimate_avg_secs_per_tick = estimate_avg_time_aggregate / estimate_avg_checked_times

                ticks_left = total_ticks - cur_ticks
                estimated_time_left = ticks_left * estimate_avg_secs_per_tick

                last_progress_report_time = time.time()
                last_progress_report_ticks = cur_tick

                print(f"{cur_tick}/{total_ticks} ({cur_tick/total_ticks:.2%}). Estimated time left: {secs_to_str(math.ceil(estimated_time_left))}")

    avg = sum_ / (len(target_words) * run_n_times)

    return avg, min_turns, max_turns, time.time() - start_t, turns__count_map

def random_algorithm_test(word):
    pc_words = [w for w in core.ALL_WORDS]
    random.shuffle(pc_words)
    
    target = word
    guess = None
    turns = 0
    while guess != target:
        turns += 1
        guess = pc_words.pop()

    return turns

def random_algorithm_test_all(guess_target_words, show_progress_every_n_secs=10, run_times=10):
    avg, min_turns, max_turns, taken_time, turns__count_map = run_all_test_inner(
        guess_target_words, random_algorithm_test,
        run_n_times=run_times, show_progress_every_n_secs=show_progress_every_n_secs
    )

    fig, ax = plt.subplots()

    min_turns_to_plot = max(0, min_turns - 1)
    max_turns_to_plot = max_turns + 1

    turns_num = max_turns_to_plot - min_turns_to_plot
    desired_ticks = 10
    turns_per_range = round(turns_num / desired_ticks)

    labels, vals = [], []
    for i in range(min_turns, max_turns + 1, turns_per_range):
        r = min(max_turns, i + turns_per_range)
        labels.append(f"{i}-{r}")

        cnt = 0
        for j in range(i, r + 1):
            cnt += turns__count_map.get(j, 0)

        vals.append(cnt)

    max_turns_count = max(vals)
    max_turns_count_to_plot = math.ceil(max_turns_count * 1.1)

    label_indexes = np.arange(len(labels))
    ax.set(xlim=(0, max_turns_count_to_plot),
           title="Turns on average (algorithm: random)", ylabel="Turns", xlabel="Count")
    ax.set_yticks(label_indexes, labels=labels)
    ax.invert_yaxis()

    bars = ax.barh(label_indexes, vals, color="blue", align="center")
    ax.bar_label(bars, fmt='%.2f')

    plt.subplots_adjust(left=0.2)

    print(f"Ran the random choice algorithm {run_times} times for each word ({len(guess_target_words)}) words. The word was guessed in {avg} turns on average. The worst game was {max_turns} turns and the best game was {min_turns}. The algorithm took {taken_time:.2f} seconds to run.")
    short_text = f"Times: {run_times} | Words: {len(guess_target_words)} | Avg: {avg:.4f} | Time: {taken_time:.2f}"
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, short_text, horizontalalignment='center')
    return fig


def naive_algorithm_test(word):
    pc_words = [w for w in core.ALL_WORDS]
    random.shuffle(pc_words)
    target = word
    guess = None
    turns = 0
    while guess != target:
        turns += 1
        guess = pc_words.pop()
        results = core.process_guess_slow(guess, target)
        pc_words = core.filter_words_slow(pc_words, guess, results)

    return turns

def naive_algorithm_test_all(guess_target_words, show_progress_every_n_secs=10, run_times=10):
    avg, min_turns, max_turns, taken_time, turns__count_map = run_all_test_inner(
        guess_target_words, naive_algorithm_test,
        run_n_times=run_times, show_progress_every_n_secs=show_progress_every_n_secs
    )

    fig, ax = plt.subplots()

    max_turns_count = max(turns__count_map.values())

    min_turns_to_plot = max(0, min_turns - 1)
    max_turns_to_plot = max_turns + 1
    max_turns_count_to_plot = math.ceil(max_turns_count * 1.1)
    ax.set(xlim=(min_turns_to_plot, max_turns_to_plot), ylim=(0, max_turns_count_to_plot),
           xticks=np.arange(min_turns, max_turns+1),

           title="Turns on average (algorithm: naive)", xlabel="Turns", ylabel="Count")

    x_vals, y_vals = [], []
    for i in range(min_turns, max_turns + 1):
        x_vals.append(i)
        y_vals.append(turns__count_map.get(i, 0))

    ax.bar(x_vals, y_vals, width=0.5, color="blue")

    ax.axvline(avg, color='red', linewidth=2)

    print(f"Ran the naive algorithm {run_times} times for each word ({len(guess_target_words)}) words. The word was guessed in {avg} turns on average. The worst game was {max_turns} turns and the best game was {min_turns}. The algorithm took {taken_time:.2f} seconds to run.")
    short_text = f"Times: {run_times} | Words: {len(guess_target_words)} | Avg: {avg:.4f} | Time: {taken_time:.2f}"

    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, short_text, horizontalalignment='center')
    return fig


def advanced_algorithm_test(target, pattern_matrix):
    guess = None
    target_index = core.WORD_TO_INDEX[target]

    word_indexes = np.array(list(range(len(core.ALL_WORDS))), dtype=np.uint32)
    h = hashlib.md5(str(core.ALL_WORDS).encode(encoding="utf8")).hexdigest()

    turns = 0
    while guess != target:
        turns += 1
        word_entropy_tuples = []
        if turns == 1 and os.path.exists(f"first_step_entropies_{h}.pickle"):
            with open(f"first_step_entropies_{h}.pickle", "rb") as f:
                word_entropy_tuples = pickle.load(f)
        else:
            for word_index in word_indexes:
                word = core.ALL_WORDS[word_index]
                p_chance_map, p_info_map = advanced.test_calculate_information(word_indexes, word, pattern_matrix)

                word_entropy_tuples.append((word, advanced.calc_entropy_fast(p_chance_map, p_info_map)))
            word_entropy_tuples = list(sorted(word_entropy_tuples, key=lambda x: x[1], reverse=True))

            if turns == 1:
                with open(f"first_step_entropies_{h}.pickle", "wb") as f:
                    pickle.dump(word_entropy_tuples, f)

        guess = word_entropy_tuples[0][0]
        guess_index = core.WORD_TO_INDEX[guess]
        pattern_ternary = pattern_matrix[guess_index, target_index]
        word_indexes = advanced.filter_words_fast(word_indexes, guess_index, pattern_ternary, pattern_matrix)

    return turns


def advanced_algorithm_test_all(target_words, show_progress_every_n_secs=10, run_times=1):
    pattern_matrix = core.get_pattern_matrix(core.ALL_WORDS)

    def test_one(word):
        return advanced_algorithm_test(word, pattern_matrix)

    avg, min_turns, max_turns, taken_time, turns__count_map = run_all_test_inner(
        target_words, test_one, run_n_times=run_times,
        show_progress_every_n_secs=show_progress_every_n_secs
    )

    fig, ax = plt.subplots()

    max_turns_count = max(turns__count_map.values())

    min_turns_to_plot = max(0, min_turns - 1)
    max_turns_to_plot = max_turns + 1
    max_turns_count_to_plot = math.ceil(max_turns_count * 1.1)
    ax.set(xlim=(min_turns_to_plot, max_turns_to_plot), ylim=(0, max_turns_count_to_plot),
           xticks=np.arange(min_turns, max_turns + 1),

           title="Turns on average (algorithm: advanced)", xlabel="Turns", ylabel="Count")

    x_vals, y_vals = [], []
    for i in range(min_turns, max_turns + 1):
        x_vals.append(i)
        y_vals.append(turns__count_map.get(i, 0))

    ax.axvline(avg, color='red', linewidth=2)
    ax.bar(x_vals, y_vals, width=0.5, color="blue")

    print(
        f"Ran the naive algorithm {run_times} times for each word ({len(target_words)}) words. The word was guessed in {avg} turns on average. The worst game was {max_turns} turns and the best game was {min_turns}." +
        f" The time taken was {taken_time:.2f} seconds."
    )
    short_text = f"Times: {run_times} | Words: {len(guess_target_words)} | Avg: {avg:.4f} | Time: {taken_time:.2f}"
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, short_text, horizontalalignment='center')

    return fig


# only use this if the word list is small enough, since this disables the first step entropy cache
DEBUG_STEP_BY_STEP = False
def advanced_algorithm_interactive_solver():
    print("Reminder: 0 means the letter is not in the word, 1 means it's in the word but in the wrong position, 2 means it's in the word and in the right position.")
    pattern_matrix = core.get_pattern_matrix(core.ALL_WORDS)

    word_indexes = np.array(list(range(len(core.ALL_WORDS))), dtype=np.uint32)
    h = hashlib.md5(str(core.ALL_WORDS).encode(encoding="utf8")).hexdigest()
    turns = 0

    while True:
        turns += 1
        word_entropy_tuples = []
        if not DEBUG_STEP_BY_STEP and turns == 1 and os.path.exists(f"first_step_entropies_{h}.pickle"):
            with open(f"first_step_entropies_{h}.pickle", "rb") as f:
                word_entropy_tuples = pickle.load(f)
        else:
            for word_index in word_indexes:
                word = core.ALL_WORDS[word_index]
                ts = time.time() * 1000
                p_chance_map, p_info_map = advanced.test_calculate_information(word_indexes, word,
                                                                               pattern_matrix)
                te = time.time() * 1000

                word_entropy_tuples.append((word, advanced.calc_entropy_fast(p_chance_map, p_info_map)))
                if DEBUG_STEP_BY_STEP:
                    print(f"{word} took {te-ts}ms")
                    pprint({pattern_to_str(ternary_pattern_to_list(k)): v for k, v in p_chance_map.items()})
                    pprint({pattern_to_str(ternary_pattern_to_list(k)):v for k,v in p_info_map.items()})
                    print(advanced.calc_entropy_fast(p_chance_map, p_info_map))
                    print("\n\n")

            word_entropy_tuples = list(sorted(word_entropy_tuples, key=lambda x: x[1], reverse=True))

            if turns == 1:
                with open(f"first_step_entropies_{h}.pickle", "wb") as f:
                    pickle.dump(word_entropy_tuples, f)

        guess = word_entropy_tuples[0][0]
        guess_index = core.WORD_TO_INDEX[guess]
        print(f"My guess is: {guess}. Expected information: {word_entropy_tuples[0][1]}. Possible words left: {len(word_indexes)}")
        # information bits to words left
        expected_words_left = math.ceil(2**(-word_entropy_tuples[0][1])*len(word_indexes))
        print(f"Expected words left after guess: {expected_words_left}")

        skip_word = False
        while True:
            pattern_ternary_raw = input("Enter the guess result (like this: 01200), or enter s to skip this word.")
            if pattern_ternary_raw == "s":
                skip_word = True
                break
            if len(pattern_ternary_raw) != len(guess) or not all(c in "012" for c in pattern_ternary_raw):
                print("Invalid input.")
                continue
            break

        if skip_word:
            word_indexes = word_indexes[word_indexes != guess_index]
            print("Skipping the guess.")
            continue

        if pattern_ternary_raw == "22222":
            print(f"Congrats! We guessed the word in {turns} turns.")
            break
        pattern_ternary = pattern_str_to_ternary_pattern(pattern_ternary_raw)
        len_before = len(word_indexes)
        word_indexes = advanced.filter_words_fast(word_indexes, guess_index, pattern_ternary, pattern_matrix)
        if len(word_indexes) == 0:
            print("No valid words left. This means either one of the results was inputted wrong, or the word is not in the list.")
            break
        len_after = len(word_indexes)
        gotten_information = - math.log2(len_after / len_before)
        print(f"We got {gotten_information} information from this guess.")
        diff = gotten_information - word_entropy_tuples[0][1]
        if diff > 0:
            print(f"We were lucky, we got {diff} more information than expected (so we got rid of {expected_words_left - len(word_indexes)} more words than expected).")
        elif diff < 0:
            print(f"We were unlucky, we got {-diff} less information than expected (so we got rid of {len(word_indexes) - expected_words_left} less words than expected).")
        else:
            print("We got the exact amount of information we expected.")


from pprint import pprint

def pattern_to_str(pattern):
    return " ".join(str(p.value) for p in pattern)

def ternary_pattern_to_list(pattern):
    result = []
    curr = pattern
    for x in range(core.WORD_LEN):
        v = curr % 3
        v = {
            0: core.GuessResults.NOT_PRESENT,
            1: core.GuessResults.PRESENT,
            2: core.GuessResults.CORRECT
        }[v]
        result.append(v)
        curr = curr // 3
    return result

def pattern_str_to_ternary_pattern(pattern_str):
    return int(np.dot(
        np.array([int(c) for c in pattern_str]),
        (3 ** np.arange(len(pattern_str))).astype(np.uint8)
    ))

import time

if __name__ == "__main__":
    guess_target_words = [word for word in core.ALL_WORDS]

    while True:
        i = input("Enter:\n* 1 if you want to play against the computer, \n* 2 if you want to guess words, \n* 3 if you want to have the PC guess your word.\n* 4 if you want to test the random algorith\n* 5 if you want to test the naive algorithm\n* 6 if you want to test the advanced algorithm.\n* 7 if you want to test all algorithms\n* 8 if you want the AI to solve an unknown word.\n>")
        if i not in ("1", "2", "3", "4", "5", "6", "7", "8"):
            print("Invalid input, try again.\n")
            continue

        if i == "4":
            fig_rand = random_algorithm_test_all(guess_target_words)
            plt.show()
            continue

        if i == "5":
            naive_algorithm_test_all(guess_target_words)
            plt.show()
            continue

        if i == "6":
            # take every Nth word, as it normally takes up to 30-60 minutes to go through 10000 games
            t = time.time()
            guess_target_words = [core.ALL_WORDS[i] for i in range(0, len(core.ALL_WORDS), 1)]
            advanced_algorithm_test_all(guess_target_words)
            plt.show()
            print(f"It took us {time.time() - t} seconds to test the advanced algorithm on {len(guess_target_words)} words.")
            exit(0)
            continue

        if i == "7":
            # test all
            # take every Nth word, as it normally takes up to 30-60 minutes to go through 10000 games
            guess_target_words = [core.ALL_WORDS[i] for i in range(0, len(core.ALL_WORDS), 1)]
            tt = time.time()
            f1 = random_algorithm_test_all(guess_target_words, run_times=10, show_progress_every_n_secs=10)
            f1.savefig(f"random_algorithm_test_all.png")
            f2 = naive_algorithm_test_all(guess_target_words, run_times=10, show_progress_every_n_secs=10)
            f2.savefig(f"naive_algorithm_test_all.png")
            f3 = advanced_algorithm_test_all(guess_target_words, run_times=1, show_progress_every_n_secs=10)
            f3.savefig(f"advanced_algorithm_test_all.png")
            print(f"It took us {time.time() - tt} seconds to test all algorithms")

            plt.show()
            continue

        if i == "8":
            advanced_algorithm_interactive_solver()
            continue

        mode = {
            "1": GameModes.VERSUS,
            "2": GameModes.PLAYER_ONLY,
            "3": GameModes.PC_ONLY
        }[i]
        break
    
    game = Game(mode=mode)
    game.play()
    




