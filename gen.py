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
                words = [line.strip() for line in f.readlines()]

        self._words = {l1: {l2: [] for l2 in string.ascii_lowercase}
                       for l1 in string.ascii_lowercase}
        self._length = 0

        for word in words:
            w = self.clean(word)
            if 2 <= len(w) <= max_length:
                self._length += 1
                self._words[w[0]][w[1]].append(w)

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
        for d in self._words.values():
            for words in d.values():
                yield from words

    def __len__(self):
        return self._length

    def __contains__(self, word):
        if len(word) < 2:
            return False
        l1, l2 = word[:2]
        return word in self._words[l1][l2]


def last_word_from_line(line):
    for c in range(len(line) - 1, -1, -1):
        if line[c] in (EMPTY, BLOCK):
            return ''.join(line[c + 1:])
    return ''.join(line)


def word_matches_vertically(wordlist, word, grid, pos):
    logger.debug('--- Checking if word=%s matches vertically at pos=%s',
                 word, pos)
    words_x = range(pos[0], pos[0] + len(word))
    y = pos[1]
    if y == 0:
        return True
    max_additional_length = grid.height - y
    for x, letter in zip(words_x, word):
        col = grid[(x, 0):(x, y - 1)]
        row = col.transpose()
        word = ''.join((last_word_from_line(row[0]), letter))
        logger.debug('checking word=%s from column %d', word, x)
        max_length = len(word) + max_additional_length

        if not any(len(w) <= max_length for w in wordlist.starting_with(word)):
            logger.debug('--- No :-(')
            # FIXME: cache correct words
            return False
    logger.debug('--- Yes X-)')
    return True


def iter_correct_words(wordlist, grid, start_pos, end_pos):
    wl = list(wordlist)
    random.shuffle(wl)
    yield from (w for w in wl if len(w) <= end_pos[0] + 1 - start_pos[0]
                and word_matches_vertically(wordlist, w, grid, start_pos))


def generate_grid(wordlist, dimensions):
    random.seed(42)
    width, height = dimensions
    grid = Grid([[EMPTY for _ in range(width)] for _ in range(height)])
    x, y = (0, 0)
    stack = []
    wl = list(wordlist)
    random.shuffle(wl)
    correct_words = iter(wl)
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
            correct_words = iter_correct_words(
                wordlist, grid, (x, y), (width - 1, y))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('=== Grid:\n%s', grid)
        else:
            print(f'\033c=== Grid:\n{grid}')
    return grid


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    wordlist_file = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    wordlist = WordList(wordlist_file, width)
    grid = generate_grid(wordlist, (width, height))
    print(grid)
