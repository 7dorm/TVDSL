"""HW2 lexer: HW1's minimized DFA with source positions and EOF."""

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache

from automata import ASCII, RULES, build


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    offset: int
    line: int
    column: int

    def position(self):
        return dict(offset=self.offset, line=self.line, column=self.column)


class LexError(ValueError):
    def __init__(self, message, position):
        self.diagnostic = dict(phase="lexical", message=message, **position)
        super().__init__(f"{position['line']}:{position['column']}: {message}")


@lru_cache(maxsize=1)
def machine():
    return build()[2]


def line_starts(text):
    starts, cursor = [0], 0
    while cursor < len(text):
        if text[cursor] == "\r":
            if text[cursor:cursor + 2] == "\r\n":
                cursor += 1
            starts.append(cursor + 1)
        elif text[cursor] == "\n":
            starts.append(cursor + 1)
        cursor += 1
    return starts


def tokenize(text):
    dfa, starts = machine(), line_starts(text)

    def location(offset):
        line = bisect_right(starts, offset)
        return dict(offset=offset, line=line, column=offset - starts[line - 1] + 1)

    tokens, position = [], 0
    while position < len(text):
        state, cursor, last = dfa.start, position, None
        while cursor < len(text):
            state = dfa.step(state, text[cursor])
            if state == dfa.trap:
                if ord(text[cursor]) not in ASCII:
                    raise LexError("Non-ASCII character", location(cursor))
                break
            cursor += 1
            if dfa.accept[state] is not None:
                last = cursor, dfa.accept[state]
        if last is None:
            raise LexError(f"Unexpected character {text[position]!r}", location(position))
        end, index = last
        rule, lexeme = RULES[index], text[position:end]
        if rule.kind == "INT" and lexeme == "0" and end < len(text) and text[end] in "0123456789":
            raise LexError("Leading zero in integer", location(position))
        if not rule.skip:
            tokens.append(Token(rule.kind, lexeme, **location(position)))
        position = end
    tokens.append(Token("EOF", "", **location(len(text))))
    return tokens
