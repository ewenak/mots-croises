#! /usr/bin/env python3

import logging
import os
import random
import re
import string
import typing
import unicodedata

logger = logging.getLogger(__name__)

BLOCK = '\N{BLACK SQUARE}'
EMPTY = '\N{WHITE SQUARE}'


class Grid:
    def __init__(self, data: typing.Iterable[typing.Iterable], dimensions=None,
                 default=None, *, error_out_of_bounds=True, _trust_args=False):
        if _trust_args:
            # Bypass args processing, as we're called from an internal method.
            # Will be faster, thanks to not doing list(r) for r in data
            self.data = data
            self.dimensions = dimensions
            self.width, self.height = dimensions
            self.error_out_of_bounds = error_out_of_bounds
            return

        if isinstance(data, Grid):
            data = data.data
        self.data = [list(r) for r in data]
        if dimensions is None:
            dimensions = (len(self.data[0]), len(self.data))
        self.width, self.height = dimensions
        self.dimensions = dimensions
        self.error_out_of_bounds = error_out_of_bounds

    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            if pos in self:
                x, y = pos
                return self.data[y][x]
            elif self.error_out_of_bounds:
                raise IndexError(
                    f'{pos} not in grid of dimensions {self.dimensions}')
            return None
        elif isinstance(pos, slice):
            x0, y0 = pos.start or (0, 0)

            if pos.stop is not None:
                x1 = pos.stop[0] + 1
                y1 = pos.stop[1] + 1
            else:
                x1, y1 = self.width, self.height

            if x1 > self.width:
                x1 = self.width
            if y1 > self.height:
                y1 = self.height

            if pos.step:
                raise NotImplementedError('slices with steps unsupported')

            return Grid(
                [self[y][x0:x1] for y in range(y0, y1)],
                dimensions=(x1 - x0, y1 - y0),
                error_out_of_bounds=self.error_out_of_bounds, _trust_args=True
            )

        return self.data[pos]

    def __setitem__(self, pos, value):
        if isinstance(pos, tuple):
            x, y = pos
            self.data[y][x] = value
        elif isinstance(pos, slice):
            x0, y0 = pos.start or (0, 0)
            x1, y1 = pos.stop or (self.width - 1, self.height - 1)
            x1 += 1
            y1 += 1
            if pos.step:
                raise NotImplementedError('slices with steps unsupported')

            if self.error_out_of_bounds and (x1 > self.width
                                             or y1 > self.height):
                raise ValueError(
                    f'slice {pos} does not fit in grid of dimensions '
                    f'{self.dimensions}'
                )
            w = x1 - x0
            h = y1 - y0

            if self.error_out_of_bounds and len(value) != h:
                raise ValueError(
                    f'value {value} is not of the right size for slice {pos}')

            for y, raw_row in zip(range(y0, y1), reversed(value)):
                row = list(raw_row)
                if len(raw_row) > w:
                    if self.error_out_of_bounds:
                        raise ValueError(
                            f'row {raw_row} does not fit in slice {pos}')
                    else:
                        row = row[:w]

                self.data[y][x0:x1] = row
        else:
            self.data[pos] = list(value)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return '\n'.join(
            ' '.join(str(i).center(4)[:5] for i in row)
            for row in self.data
        )

    def __repr__(self):
        import pprint
        lines = pprint.pformat(self.data).splitlines()
        if len(lines) == 1:
            return f'{self.__class__.__qualname__}({lines[0]})'
        else:
            f = '\n'.join(f'    {line}' for line in lines)
            return f'{self.__class__.__qualname__}(\n{f}\n)'

    def __contains__(self, point):
        x, y = point
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height

    def set_if_not_none(self, pos, value):
        if pos not in self:
            if self.error_out_of_bounds:
                raise ValueError(
                    f'pos {pos} not in grid of dimensions {self.dimensions}')
            else:
                return

        if self[pos] is not None:
            self[pos] = value

    def transpose(self):
        return Grid(
            [list(r) for r in zip(*self.data)],
            dimensions=(self.height, self.width),
            error_out_of_bounds=self.error_out_of_bounds, _trust_args=True
        )

    def neighbors(self, pos):
        x, y = pos
        return [
            self[c, r]
            for c, r in ((x - 1, y), (x + 1, y),
                         (x, y - 1), (x, y + 1))
            if (c, r) in self and self[c, r] is not None
        ]


class WordList:
    def __init__(self, words: os.PathLike | str | list, max_length=8):
        if isinstance(words, (os.PathLike, str)):
            with open(words, "r") as f:
                words = [
                    word for line in f.readlines()
                    if 2 <= len(word := self.clean(line.strip())) <= max_length
                ]

        random.shuffle(words)
        self._length = len(words)

        self._words = {l1: {l2: [] for l2 in string.ascii_lowercase}
                       for l1 in string.ascii_lowercase}
        self._word_list = words

        for word in words:
            self._words[word[0]][word[1]].append(word)

    @staticmethod
    def clean(word):
        """Clean word, so that it only contains unaccented latin letters"""
        # Remove accents
        word = unicodedata.normalize('NFD', word)
        word = word.encode('ascii', 'ignore')
        word = word.decode("utf-8")
        word = word.lower()
        word, _ = re.subn(r'[^a-z]', '', word)

        return word

    def starting_with(self, start: str) -> list[str]:
        if len(start) == 0:
            return list(self)
        elif len(start) == 1:
            return list(self._words[start])
        l1, l2 = start[:2]
        return [w for w in self._words[l1][l2] if w.startswith(start)]

    def __iter__(self):
        return iter(self._word_list)

    def __len__(self):
        return self._length

    def __contains__(self, word):
        if len(word) < 2:
            return False
        l1, l2 = word[:2]
        return word in self._words[l1][l2]


class CorrectWordsFinder:
    def __init__(self, wordlist, grid, start_pos, end_pos,
                 columns_last_word=None, columns_possible_words=None):
        self.wordlist = wordlist
        self.grid = grid
        self.start_pos = start_pos
        self.end_pos = end_pos
        if columns_last_word is not None:
            self.columns_last_word = columns_last_word
        else:
            self.columns_last_word = self.get_columns_last_word()
        if columns_possible_words is not None:
            self.columns_possible_words = columns_possible_words
        else:
            self.columns_possible_words = [
                list(wordlist) for _ in range(grid.width)
            ]

    def get_columns_last_word(self):
        columns_last_word = []
        for col in self.grid.transpose():
            word = []
            for char in col:
                if char == EMPTY:
                    break
                elif char == BLOCK:
                    word.clear()
                else:
                    word.append(char)
            columns_last_word.append(''.join(word))
        return columns_last_word

    def find_solution(self):
        if self.start_pos[1] >= self.grid.height:
            return self.grid

        for w in self.wordlist:
            if len(w) <= self.end_pos[0] + 1 - self.start_pos[0]:
                x, y = self.start_pos
                possible_words = self.list_column_possible_words_match(w)
                if possible_words is not None:
                    end_x = x + len(w) - 1
                    logger.debug('x=%d y=%d end_x=%d word=%s', x, y, end_x, w)
                    grid = Grid(self.grid)
                    grid[(x, y):(end_x, y)] = (w,)
                    columns_last_word = []
                    for i in range(grid.width):
                        if x <= i <= end_x:
                            columns_last_word.append(''.join(
                                (self.columns_last_word[i], w[i - x])))
                        else:
                            columns_last_word.append(self.columns_last_word[i])
                    if end_x < grid.width - 1:
                        # Adding BLOCK on next cell and advancing two cells
                        grid[(end_x + 1, y)] = BLOCK
                        x = end_x + 2
                        columns_last_word[end_x] = ''
                    else:
                        # end_x == width - 1, the word ends the line, let's
                        # continue with the next line
                        x = 0
                        y += 1
                        logger.debug('=== Going to next line')
                    print(grid)
                    breakpoint()
                    solved_grid = CorrectWordsFinder(
                        self.wordlist, grid, (x, y), (grid.width - 1, y),
                        columns_last_word, possible_words).find_solution()
                    if solved_grid is not None:
                        return solved_grid
        return None

    def list_column_possible_words_match(self, horizontal_word):
        #logger.debug(
        #    '--- Checking if word=%s matches vertically at pos=%s',
        #    horizontal_word, self.start_pos
        #)
        x = self.start_pos[0]
        max_additional_length = self.grid.height - self.start_pos[1]
        columns_possible_words = list(self.columns_possible_words)
        for letter, column_word, i in zip(
            horizontal_word,
            self.columns_last_word[x : x+len(horizontal_word)],
            range(x, x + len(horizontal_word))
        ):
            new_word = ''.join((column_word, letter))
            max_length = len(new_word) + max_additional_length
            columns_possible_words[i] = [
                w for w in columns_possible_words[i]
                if w.startswith(new_word) and len(w) < max_length
            ]
            if len(columns_possible_words[i]) == 0:
                #logger.debug('--- No :-(')
                return None
        #logger.debug('--- Yes X-)')
        return columns_possible_words


def generate_grid(wordlist, dimensions):
    width, height = dimensions
    grid = Grid([[EMPTY for _ in range(width)] for _ in range(height)])
    x, y = (0, 0)
    stack = []
    correct_words = iter(wordlist)
    logger.debug('=== Grid:\n%s', grid)
    while y < height:
        try:
            word = next(correct_words)
        except StopIteration:
            logger.debug('=== Going back to previous state')
            grid, correct_words, (x, y) = stack.pop()
        else:
            logger.debug('=== Trying word: %s', word)
            stack.append((Grid(grid), correct_words, (x, y)))
            end_x = x + len(word) - 1
            logger.debug('x=%d y=%d end_x=%d word=%s', x, y, end_x, word)
            grid[(x, y):(end_x, y)] = (word,)
            if end_x < width - 1:
                # Adding BLOCK on next cell and advancing two cells
                grid[(end_x + 1, y)] = BLOCK
                x = end_x + 2
            else:
                # end_x == width - 1, the word ends the line, let's continue
                # with the next line
                x = 0
                y += 1
                logger.debug('=== Going to next line')
            #correct_words = iter_correct_words(
            #    wordlist, grid, (x, y), (width - 1, y))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('=== Grid:\n%s', grid)
        else:
            #print(f'\033c=== Grid:\n{grid}')
            print(len(stack))
    return grid


if __name__ == '__main__':
    #import sys
    logging.basicConfig(level=logging.INFO)
    #wordlist_file = sys.argv[1]
    #width = int(sys.argv[2])
    #height = int(sys.argv[3])
    if __debug__:
        random.seed(42)
    #wordlist = WordList(wordlist_file, width)
    #grid = generate_grid(wordlist, (width, height))
    print('=== Grid:')
    #print(grid)
