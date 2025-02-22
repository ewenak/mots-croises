#! /usr/bin/env python3

import os
import random
import string
import unicodedata
import re
import typing


EMPTY_CELL = object()    # FIXME: use a better sentinel object


class Grid:
    def __init__(self, data: typing.Iterable[typing.Iterable], default=None):
        if isinstance(data, Grid):
            data = data.data
        self.data = [list(r) for r in data]

    @property
    def width(self):
        return len(self.data[0])

    @property
    def height(self):
        return len(self.data)

    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            x, y = pos
            if 0 <= x < self.width and 0 <= y < self.height:
                return self.data[y][x]
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

            return Grid(self[y][x0:x1] for y in range(y0, y1))

        return self.data[pos]

    def __setitem__(self, pos, value):
        if isinstance(pos, tuple):
            x, y = pos
            self.data[y][x] = value
        elif isinstance(pos, slice):
            x0, y0 = pos.start or (0, 0)
            x1, y1 = pos.stop or (0, 0)
            x1 += 1
            y1 += 1
            if pos.step:
                raise NotImplementedError('slices with steps unsupported')

            for y, i in zip(range(y0, y1), range(y1 - y0)):
                self.data[y][x0:x1] = value[i]
        else:
            self.data[pos] = value

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

    def set_if_not_none(self, pos, value):
        if self[pos] is not None:
            self[pos] = value

    def transpose(self):
        return Grid(list(zip(*self.data)))

    def neighbors(self, pos):
        x, y = pos
        return [
            self[c, r]
            for c, r in ((x - 1, y), (x + 1, y),
                         (x, y - 1), (x, y + 1))
            if self[c, r] is not None
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
            return [w for d in self._words[start] for w in d.values()]
        l1, l2 = start[:2]
        return [w for w in self._words[l1][l2] if w.startswith(start)]

    def __iter__(self):
        for d in self._words.values():
            for words in d.values():
                yield from words

    def __len__(self):
        return self._length


def last_word_from_line(line):
    for c in range(len(line) - 1, -1, -1):
        if line[c] is EMPTY_CELL or line[c] is None:
            return ''.join(line[c + 1:])
    return ''.join(line)


def word_matches_vertically(wordlist, word, grid, pos):
    words_x = range(pos[0], pos[0] + len(word) - 1)
    y = pos[1]
    if y == 0:
        return True
    max_additional_length = grid.height - y
    for x, letter in zip(words_x, word):
        col = grid[(x, 0):(x, y - 1)]
        row = col.transpose()
        word = ''.join((last_word_from_line(row[0]), letter))
        max_length = len(word) + max_additional_length

        if not any(len(w) <= max_length for w in wordlist.starting_with(word)):
            # FIXME: cache correct words
            return False
    return True


def list_correct_words(wordlist, grid, start_pos, end_pos):
    wl = list(wordlist)
    random.shuffle(wl)
    yield from (w for w in wl if len(w) <= end_pos[0] + 1 - start_pos[0]
                and word_matches_vertically(wordlist, w, grid, start_pos))
