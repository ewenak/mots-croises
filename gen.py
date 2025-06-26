#! /usr/bin/env python3

import logging
import os
import random
import re
import string
import sys
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


class SolveAttempt:
    def __init__(self, wordlist, grid, x=0, y=0):
        self.wordlist = wordlist
        self.grid = grid
        self.x = x
        self.y = y
        self.correct_words = self.iter_correct_words()

    def last_word_from_line(self, line):
        for c in range(len(line) - 1, -1, -1):
            if line[c] in (EMPTY, BLOCK):
                return ''.join(line[c + 1:])
        return ''.join(line)

    def word_matches_vertically(self, word):
        logger.debug('--- Checking if word=%s matches at pos=%s',
                     word, (self.x, self.y))
        words_x = range(self.x, self.x + len(word))
        if self.y == 0:
            return True
        max_additional_length = self.grid.height - self.y
        for x, letter in zip(words_x, word):
            col = self.grid[(x, 0):(x, self.y - 1)]
            row = col.transpose()
            word = ''.join((self.last_word_from_line(row[0]), letter))
            logger.debug('checking word=%s from column %d', word, x)
            max_length = len(word) + max_additional_length

            if not any(
                len(w) <= max_length
                for w in self.wordlist.starting_with(word)
            ):
                logger.debug('--- No :-(')
                # FIXME: cache correct words
                return False
        logger.debug('--- Yes X-)')
        return True

    def iter_correct_words(self):
        yield from (
            w for w in self.wordlist
            if (len(w) <= self.grid.width - self.x
                and self.word_matches_vertically(w))
        )

    def recurse(self):
        if self.y >= self.grid.height:
            return self.grid
        for word in self.correct_words:
            logger.debug('Trying word: %s', word)
            new = self.copy()
            new.add_word(word)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('=== Grid:\n%s', new.grid)
            else:
                print(f'\033c=== Grid:\n{new.grid}', file=sys.stderr)
            grid = new.recurse()
            if grid is not None:
                return grid
            logger.debug('Going back to previous state')
        return None

    def add_word(self, word):
        end_x = self.x + len(word) - 1
        logger.debug('x=%d y=%d end_x=%d word=%s', self.x, self.y, end_x, word)
        self.grid[(self.x, self.y):(end_x, self.y)] = (word,)
        if end_x < self.grid.width - 1:
            # Adding BLOCK on next cell and advancing two cells
            self.grid[(end_x + 1, self.y)] = BLOCK
            self.x = end_x + 2
        else:
            # end_x == width - 1, the word ends the line, let's continue
            # with the next line
            self.x = 0
            self.y += 1
            logger.debug('=== Going to next line')

    def copy(self):
        return SolveAttempt(self.wordlist, Grid(self.grid), x=self.x, y=self.y)


def generate_grid(wordlist, width, height):
    attempt = SolveAttempt(
        wordlist,
        Grid([[EMPTY for _ in range(width)] for _ in range(height)])
    )
    return attempt.recurse()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Crossword generator')
    parser.add_argument(
        'wordlist', help='Wordlist file path (1 word per line)')
    parser.add_argument('width', type=int, help='Grid width')
    parser.add_argument('height', type=int, help='Grid height')
    parser.add_argument('--debug', '-d', action=argparse.BooleanOptionalAction,
                        help='Set logging level to logging.DEBUG')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    random.seed(42)
    wordlist = WordList(args.wordlist, args.width)
    grid = generate_grid(wordlist, args.width, args.height)
    print(grid)
