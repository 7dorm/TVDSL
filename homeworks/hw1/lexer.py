"""HW1: Thompson NFA -> subset DFA -> Hopcroft DFA for Funny tokens."""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ASCII = frozenset(range(128))
KEYWORDS = (
    "function", "returns", "uses", "requires", "ensures", "while", "if",
    "else", "assert", "assume", "invariant", "length", "int", "true", "false",
    "not", "and", "or", "forall", "exists",
)
SYMBOLS = {
    "(": "LPAREN", ")": "RPAREN", "[": "LBRACKET", "]": "RBRACKET",
    "{": "LBRACE", "}": "RBRACE", ",": "COMMA", ";": "SEMICOLON",
    ":": "COLON", "|": "PIPE", "+": "PLUS", "-": "MINUS", "*": "STAR",
    "/": "SLASH", "==": "EQ", "!=": "NE", "<=": "LE", ">=": "GE",
    "<": "LT", ">": "GT", "=": "ASSIGN", "->": "IMPLIES", "=>": "ARROW",
}


def literal_regex(text):
    return "".join("\\" + char if char in r"\|()*+?[]" else char for char in text)


@dataclass(frozen=True)
class Rule:
    kind: str
    pattern: str
    skip: bool = False


RULES = (
    *(Rule("TYPE_INT" if word == "int" else word.upper(), word) for word in KEYWORDS),
    Rule("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    Rule("INT", r"0|[1-9][0-9]*"),
    Rule("WS", r"[ \t\r\n]+", True),
    Rule("COMMENT", r"//[^\r\n]*", True),
    *(Rule(kind, literal_regex(symbol)) for symbol, kind in SYMBOLS.items()),
)


class NFA:
    def __init__(self):
        self.edges = []
        self.accept = {}
        self.start = self.state()

    def state(self):
        self.edges.append(defaultdict(set))
        return len(self.edges) - 1

    def edge(self, source, label, target):
        self.edges[source][label].add(target)

    def closure(self, states):
        result = set(states)
        pending = list(result)
        while pending:
            for target in self.edges[pending.pop()].get(None, ()):
                if target not in result:
                    result.add(target)
                    pending.append(target)
        return frozenset(result)


class RegexParser:
    """Recursive descent: union -> concatenation -> repetition -> atom."""

    def __init__(self, pattern, nfa):
        self.pattern, self.nfa, self.pos = pattern, nfa, 0

    def peek(self):
        return self.pattern[self.pos:self.pos + 1]

    def take(self):
        char = self.peek()
        if not char:
            raise ValueError("Unexpected end of regex")
        self.pos += 1
        return char

    def fragment(self, chars):
        start, end = self.nfa.state(), self.nfa.state()
        for char in chars:
            self.nfa.edge(start, char, end)
        return start, end

    def parse(self):
        result = self.union()
        if self.peek():
            raise ValueError(f"Unexpected regex character at {self.pos}")
        return result

    def union(self):
        left = self.concat()
        while self.peek() == "|":
            self.take()
            right = self.concat()
            start, end = self.nfa.state(), self.nfa.state()
            for branch in (left, right):
                self.nfa.edge(start, None, branch[0])
                self.nfa.edge(branch[1], None, end)
            left = start, end
        return left

    def concat(self):
        parts = []
        while self.peek() and self.peek() not in "|)":
            parts.append(self.repeat())
        if not parts:
            start, end = self.nfa.state(), self.nfa.state()
            self.nfa.edge(start, None, end)
            return start, end
        for left, right in zip(parts, parts[1:]):
            self.nfa.edge(left[1], None, right[0])
        return parts[0][0], parts[-1][1]

    def repeat(self):
        inner = self.atom()
        if self.peek() and self.peek() in "*+?":
            operator = self.take()
            start, end = self.nfa.state(), self.nfa.state()
            self.nfa.edge(start, None, inner[0])
            self.nfa.edge(inner[1], None, end)
            if operator in "*?":
                self.nfa.edge(start, None, end)
            if operator in "*+":
                self.nfa.edge(inner[1], None, inner[0])
            inner = start, end
        return inner

    def character(self):
        char = self.take()
        if char == "\\":
            char = {"t": "\t", "r": "\r", "n": "\n"}.get(self.peek(), self.peek())
            self.take()
        if ord(char) not in ASCII:
            raise ValueError("Regex must use ASCII characters")
        return ord(char)

    def atom(self):
        if self.peek() == "(":
            self.take()
            inner = self.union()
            if self.take() != ")":
                raise ValueError("Expected closing parenthesis")
            return inner
        if self.peek() == "[":
            self.take()
            negate = self.peek() == "^"
            if negate:
                self.take()
            chars = set()
            while self.peek() and self.peek() != "]":
                first = self.character()
                if self.peek() == "-" and self.pattern[self.pos + 1:self.pos + 2] != "]":
                    self.take()
                    last = self.character()
                    if first > last:
                        raise ValueError("Reversed character range")
                    chars.update(range(first, last + 1))
                else:
                    chars.add(first)
            if self.take() != "]" or not chars:
                raise ValueError("Expected nonempty character class")
            return self.fragment(ASCII - chars if negate else chars)
        if self.peek() and self.peek() in "|)*+?]":
            raise ValueError(f"Unexpected regex operator at {self.pos}")
        return self.fragment({self.character()})


def build_nfa(rules=RULES):
    nfa = NFA()
    for priority, rule in enumerate(rules):
        start, end = RegexParser(rule.pattern, nfa).parse()
        if end in nfa.closure({start}):
            raise ValueError(f"Token {rule.kind} can match an empty string")
        nfa.edge(nfa.start, None, start)
        nfa.accept[end] = priority
    return nfa


@dataclass
class DFA:
    transitions: list[list[int]]
    accept: list[int | None]
    start: int
    trap: int

    def step(self, state, char):
        code = ord(char)
        return self.transitions[state][code] if code in ASCII else self.trap


def determinize(nfa):
    initial = nfa.closure({nfa.start})
    subsets = [initial, frozenset()]
    indexes = {initial: 0, frozenset(): 1}
    transitions, accept = [], []
    for subset in subsets:
        winners = [nfa.accept[state] for state in subset if state in nfa.accept]
        accept.append(min(winners) if winners else None)
        row = []
        for code in range(128):
            targets = set()
            for state in subset:
                targets.update(nfa.edges[state].get(code, ()))
            target = nfa.closure(targets)
            if target not in indexes:
                indexes[target] = len(subsets)
                subsets.append(target)
            row.append(indexes[target])
        transitions.append(row)
    return DFA(transitions, accept, 0, indexes[frozenset()])


def minimize(dfa):
    groups = defaultdict(set)
    for state, output in enumerate(dfa.accept):
        groups[output].add(state)
    partitions = list(groups.values())
    pending = [group.copy() for group in partitions]
    reverse = [defaultdict(set) for _ in range(128)]
    for source, row in enumerate(dfa.transitions):
        for code, target in enumerate(row):
            reverse[code][target].add(source)
    while pending:
        splitter = pending.pop()
        for code in range(128):
            predecessors = set()
            for target in splitter:
                predecessors.update(reverse[code].get(target, ()))
            refined = []
            for group in partitions:
                inside, outside = group & predecessors, group - predecessors
                if not inside or not outside:
                    refined.append(group)
                    continue
                refined.extend((inside, outside))
                if group in pending:
                    pending.remove(group)
                    pending.extend((inside, outside))
                else:
                    pending.append(inside if len(inside) <= len(outside) else outside)
            partitions = refined
    partitions.sort(key=min)
    owner = {state: index for index, group in enumerate(partitions) for state in group}
    transitions = [[owner[target] for target in dfa.transitions[min(group)]] for group in partitions]
    accept = [dfa.accept[min(group)] for group in partitions]
    return DFA(transitions, accept, owner[dfa.start], owner[dfa.trap])


def build(rules=RULES):
    nfa = build_nfa(rules)
    dfa = determinize(nfa)
    return nfa, dfa, minimize(dfa)


class LexError(ValueError):
    def __init__(self, position, message):
        self.position = position
        super().__init__(f"Offset {position}: {message}")


def tokenize(text, dfa, rules=RULES):
    tokens, position = [], 0
    while position < len(text):
        state, cursor, last = dfa.start, position, None
        while cursor < len(text):
            state = dfa.step(state, text[cursor])
            if state == dfa.trap:
                if ord(text[cursor]) not in ASCII:
                    raise LexError(cursor, "Non-ASCII character")
                break
            cursor += 1
            if dfa.accept[state] is not None:
                last = cursor, dfa.accept[state]
        if last is None:
            raise LexError(position, f"Unexpected character {text[position]!r}")
        end, rule_index = last
        rule, lexeme = rules[rule_index], text[position:end]
        if rule.kind == "INT" and lexeme == "0" and end < len(text) and text[end] in "0123456789":
            raise LexError(position, "Leading zero in integer")
        if not rule.skip:
            tokens.append((rule.kind, lexeme))
        position = end
    return tokens


def serialize(dfa, rules=RULES):
    return {
        "format": "funny-dfa-v1", "alphabet_codes": list(range(128)),
        "start": dfa.start, "trap": dfa.trap, "outside_ascii": dfa.trap,
        "rules": [dict(kind=rule.kind, pattern=rule.pattern, skip=rule.skip) for rule in rules],
        "accept": dfa.accept, "transitions": dfa.transitions,
        "scanner_policy": {"longest_match": True, "tie_break": "first_rule", "reject_leading_zero": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Tokenize this string")
    parser.add_argument("--file", type=Path, help="Tokenize a UTF-8 file")
    parser.add_argument("--export", nargs="?", const="-", help="Export DFA as JSON (default: stdout)")
    args = parser.parse_args()
    if args.text is not None and args.file is not None:
        parser.error("Choose either --text or --file")
    if args.export == "-" and (args.text is not None or args.file is not None):
        parser.error("Export to a file when also tokenizing input")
    nfa, dfa, minimal = build()
    if args.export is not None:
        payload = json.dumps(serialize(minimal), ensure_ascii=True, separators=(",", ":")) + "\n"
        if args.export == "-":
            print(payload, end="")
        else:
            Path(args.export).write_text(payload, encoding="utf-8")
    if args.text is not None or args.file is not None:
        try:
            text = args.text if args.text is not None else args.file.read_bytes().decode("utf-8")
            print(json.dumps(tokenize(text, minimal), ensure_ascii=True))
        except (LexError, OSError, UnicodeError) as error:
            parser.exit(1, f"{error}\n")
    elif args.export is None:
        print(f"NFA: {len(nfa.edges)}; DFA: {len(dfa.transitions)}; minimal DFA: {len(minimal.transitions)}")
        print(f"Start: {minimal.start}; trap: {minimal.trap}; rules: {len(RULES)}; alphabet: 128 ASCII codes")


if __name__ == "__main__":
    main()
