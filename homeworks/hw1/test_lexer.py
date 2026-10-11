"""CLI: python -B -m unittest -v test_lexer.py (no generated files)."""

import itertools
import json
import random
import re
import subprocess
import sys
import unittest
from collections import deque
from pathlib import Path

from lexer import (
    ASCII, KEYWORDS, RULES, SYMBOLS, DFA, LexError, Rule, build,
    build_nfa, serialize, tokenize,
)


POSITIVE = [
    ("empty", "", []),
    ("whitespace", " \t\r\n\r\n  ", []),
    ("zero", "0", [("INT", "0")]),
    ("integers", "1 42 1234567890", [("INT", "1"), ("INT", "42"), ("INT", "1234567890")]),
    ("underscores", "_ _a a_b2 A0", [("IDENT", "_"), ("IDENT", "_a"), ("IDENT", "a_b2"), ("IDENT", "A0")]),
    ("keyword_prefix", "if ifx if_ function2", [("IF", "if"), ("IDENT", "ifx"), ("IDENT", "if_"), ("IDENT", "function2")]),
    ("case_sensitive", "If FUNCTION", [("IDENT", "If"), ("IDENT", "FUNCTION")]),
    ("line_comment_eof", "// comment", []),
    ("empty_comment", "//", []),
    ("line_comment_lf", "x// hi\ny", [("IDENT", "x"), ("IDENT", "y")]),
    ("line_comment_crlf", "x// hi\r\ny", [("IDENT", "x"), ("IDENT", "y")]),
    ("line_comment_cr", "x// hi\ry", [("IDENT", "x"), ("IDENT", "y")]),
    ("comment_symbols", "// @ ! 01 [] /=\n42", [("INT", "42")]),
    ("division", "x/2// remainder", [("IDENT", "x"), ("SLASH", "/"), ("INT", "2")]),
    ("adjacent_operators", "===!==><=->", [("EQ", "=="), ("ASSIGN", "="), ("NE", "!="), ("ARROW", "=>"), ("LE", "<="), ("IMPLIES", "->")]),
    ("signed_number", "-42", [("MINUS", "-"), ("INT", "42")]),
    ("number_boundary", "12abc 0x", [("INT", "12"), ("IDENT", "abc"), ("INT", "0"), ("IDENT", "x")]),
    ("separated_zeros", "0 0", [("INT", "0"), ("INT", "0")]),
    ("program", "function f(x:int) returns r:int {r=x+1;}", [
        ("FUNCTION", "function"), ("IDENT", "f"), ("LPAREN", "("),
        ("IDENT", "x"), ("COLON", ":"), ("TYPE_INT", "int"), ("RPAREN", ")"),
        ("RETURNS", "returns"), ("IDENT", "r"), ("COLON", ":"), ("TYPE_INT", "int"),
        ("LBRACE", "{"), ("IDENT", "r"), ("ASSIGN", "="), ("IDENT", "x"),
        ("PLUS", "+"), ("INT", "1"), ("SEMICOLON", ";"), ("RBRACE", "}"),
    ]),
]
NEGATIVE = [
    ("double_zero", "00", 0), ("leading_zero", "01", 0),
    ("long_leading_zero", "00042", 0), ("leading_zero_after_token", "x=01;", 2),
    ("signed_leading_zero", "-01", 1), ("unknown", "@", 0),
    ("standalone_bang", "!", 0), ("bang_before_identifier", "!x", 0),
    ("unknown_after_token", "foo @ 42", 4), ("nul", "\0", 0),
    ("vertical_tab", "\v", 0), ("form_feed", "\f", 0), ("del", "\x7f", 0),
    ("cyrillic", "я", 0), ("unicode_after_identifier", "fooя", 3),
    ("unicode_in_comment", "// я", 3), ("unicode_after_cr", "//x\rя", 4),
    ("unicode_space", "\u00a0", 0), ("unicode_digit", "١", 0),
    ("unicode_arrow", "→", 0), ("emoji", "😀", 0),
]


def reference_tokens(text):
    """Independent test oracle; Python re is never used by the implementation."""
    if not text.isascii():
        raise ValueError("Non-ASCII")
    patterns = [re.compile(rule.pattern, re.ASCII) for rule in RULES]
    result, position = [], 0
    while position < len(text):
        matches = [(match.end(), -index) for index, pattern in enumerate(patterns)
                   if (match := pattern.match(text, position)) is not None]
        if not matches:
            raise ValueError("No token")
        end, negative_index = max(matches)
        rule, lexeme = RULES[-negative_index], text[position:end]
        if rule.kind == "INT" and lexeme == "0" and end < len(text) and text[end] in "0123456789":
            raise ValueError("Leading zero")
        if not rule.skip:
            result.append((rule.kind, lexeme))
        position = end
    return result


class LexerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nfa, cls.raw, cls.minimal = build()

    def test_positive_cases(self):
        for name, text, expected in POSITIVE:
            for label, machine in (("DFA", self.raw), ("minimal", self.minimal)):
                with self.subTest(case=name, input=text, machine=label):
                    self.assertEqual(tokenize(text, machine), expected)

    def test_negative_cases_and_offsets(self):
        for name, text, position in NEGATIVE:
            for label, machine in (("DFA", self.raw), ("minimal", self.minimal)):
                with self.subTest(case=name, input=text, machine=label):
                    with self.assertRaises(LexError) as caught:
                        tokenize(text, machine)
                    self.assertEqual(caught.exception.position, position)

    def test_all_keywords_and_boundaries(self):
        for word in KEYWORDS:
            with self.subTest(keyword=word):
                kind = "TYPE_INT" if word == "int" else word.upper()
                self.assertEqual(tokenize(word, self.minimal), [(kind, word)])
                self.assertEqual(tokenize(word + "_1", self.minimal), [("IDENT", word + "_1")])

    def test_all_symbols(self):
        for symbol, kind in SYMBOLS.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(tokenize(symbol, self.minimal), [(kind, symbol)])
        text = "()[]{},; + - * / == != <= >= < > = : | -> =>"
        expected = [(SYMBOLS[symbol], symbol) for symbol in [
            "(", ")", "[", "]", "{", "}", ",", ";", "+", "-", "*", "/",
            "==", "!=", "<=", ">=", "<", ">", "=", ":", "|", "->", "=>",
        ]]
        self.assertEqual(tokenize(text, self.minimal), expected)

    def test_every_ascii_code(self):
        for code in range(128):
            char = chr(code)
            with self.subTest(code=code):
                if char in " \t\r\n":
                    expected = []
                elif char.isascii() and (char.isalpha() or char == "_"):
                    expected = [("IDENT", char)]
                elif char in "0123456789":
                    expected = [("INT", char)]
                elif char in SYMBOLS:
                    expected = [(SYMBOLS[char], char)]
                else:
                    for machine in (self.raw, self.minimal):
                        with self.assertRaises(LexError):
                            tokenize(char, machine)
                    continue
                for machine in (self.raw, self.minimal):
                    self.assertEqual(tokenize(char, machine), expected)

    def test_every_ascii_code_in_comment(self):
        for code in range(128):
            with self.subTest(code=code):
                self.assertEqual(tokenize("//" + chr(code), self.minimal), [])

    def test_total_transition_table_and_trap(self):
        for machine in (self.raw, self.minimal):
            self.assertIsNone(machine.accept[machine.trap])
            self.assertEqual(machine.transitions[machine.trap], [machine.trap] * 128)
            for state, row in enumerate(machine.transitions):
                self.assertEqual(len(row), 128)
                self.assertTrue(all(0 <= target < len(machine.transitions) for target in row))
                for char in ("я", "\u0080", "→", "😀"):
                    self.assertEqual(machine.step(state, char), machine.trap)

    def test_exact_equivalence_of_dfas(self):
        pending, seen = deque([(self.raw.start, self.minimal.start)]), set()
        while pending:
            pair = pending.popleft()
            if pair in seen:
                continue
            seen.add(pair)
            left, right = pair
            self.assertEqual(self.raw.accept[left], self.minimal.accept[right])
            pending.extend(zip(self.raw.transitions[left], self.minimal.transitions[right]))

    def test_minimality_by_independent_moore_refinement(self):
        machine = self.minimal
        classes = machine.accept
        while True:
            signatures = [(machine.accept[state], tuple(classes[target] for target in row))
                          for state, row in enumerate(machine.transitions)]
            indexes = {}
            refined = [indexes.setdefault(signature, len(indexes)) for signature in signatures]
            if refined == classes:
                break
            classes = refined
        self.assertEqual(len(set(classes)), len(machine.transitions))
        self.assertLess(len(machine.transitions), len(self.raw.transitions))

    def test_reachability(self):
        for machine in (self.raw, self.minimal):
            pending, seen = [machine.start], set()
            while pending:
                state = pending.pop()
                if state not in seen:
                    seen.add(state)
                    pending.extend(machine.transitions[state])
            self.assertEqual(len(seen), len(machine.transitions))

    def test_serialization_round_trip(self):
        data = json.loads(json.dumps(serialize(self.minimal)))
        restored = DFA(data["transitions"], data["accept"], data["start"], data["trap"])
        rules = tuple(Rule(**rule) for rule in data["rules"])
        self.assertEqual(data["alphabet_codes"], list(range(128)))
        self.assertEqual(data["outside_ascii"], restored.trap)
        for _, text, expected in POSITIVE:
            self.assertEqual(tokenize(text, restored, rules), expected)
        for _, text, position in NEGATIVE:
            with self.assertRaises(LexError) as caught:
                tokenize(text, restored, rules)
            self.assertEqual(caught.exception.position, position)

    def test_against_regex_oracle(self):
        randomizer = random.Random(2026)
        fragments = ["if", "ifx", "_a", "0", "00", "01", "42", " ", "\t", "\r", "\n",
                     "// hi", "!=", "!", "@", "->", "=>", "a", "int", "я", *SYMBOLS]
        samples = ["".join(randomizer.choices(fragments, k=randomizer.randrange(12))) for _ in range(500)]
        for text in samples:
            with self.subTest(input=text):
                try:
                    expected = reference_tokens(text)
                except ValueError:
                    with self.assertRaises(LexError):
                        tokenize(text, self.minimal)
                else:
                    self.assertEqual(tokenize(text, self.minimal), expected)

    def test_course_examples_respect_ascii_policy(self):
        examples = sorted((Path(__file__).resolve().parents[2] / "project" / "examples").rglob("*.funny"))
        self.assertTrue(examples)
        for file in examples:
            with self.subTest(file=file.name):
                text = file.read_bytes().decode("utf-8")
                if text.isascii():
                    self.assertEqual(tokenize(text, self.minimal), reference_tokens(text))
                else:
                    first_unicode = next(index for index, char in enumerate(text) if ord(char) not in ASCII)
                    with self.assertRaises(LexError) as caught:
                        tokenize(text, self.minimal)
                    self.assertEqual(caught.exception.position, first_unicode)

    def test_cli_output_export_and_errors(self):
        def run(*args):
            return subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("lexer.py")), *args],
                                  capture_output=True, text=True, check=False)
        result = run("--text", "x=42;")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [["IDENT", "x"], ["ASSIGN", "="], ["INT", "42"], ["SEMICOLON", ";"]])
        result = run("--export")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), serialize(self.minimal))
        result = run("--file", str(Path(__file__).resolve().parents[2] / "project" / "examples" / "step01" / "good_basic.funny"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout))
        result = run("--text", "01")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Offset 0: Leading zero", result.stderr)
        result = run("--text", "x", "--export")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Export to a file", result.stderr)


class ConstructionTests(unittest.TestCase):
    def test_thompson_subset_and_hopcroft_against_re(self):
        patterns = ["a(b|c)d", "abd|acd", "(ab)*c?", "a+", "a?b", "(a|)b",
                    "[a-c]+", "[^ab]", r"[\t\r\n]+", r"\(a\)", "[a-]", "[-a]", "(a*)*b"]
        alphabet = "abcd-()\t\r\n"
        words = [""] + ["".join(chars) for length in range(1, 4)
                        for chars in itertools.product(alphabet, repeat=length)]
        for pattern in patterns:
            nfa, raw, minimal = build((Rule("TEST", "(" + pattern + ")z"),))
            for word in words:
                text = word + "z"
                expected = bool(re.fullmatch("(" + pattern + ")z", text, re.ASCII))
                states = nfa.closure({nfa.start})
                for char in text:
                    targets = set().union(*(nfa.edges[state].get(ord(char), set()) for state in states))
                    states = nfa.closure(targets)
                with self.subTest(pattern=pattern, input=text):
                    self.assertEqual(any(state in nfa.accept for state in states), expected)
                    for machine in (raw, minimal):
                        state = machine.start
                        for char in text:
                            state = machine.step(state, char)
                        self.assertEqual(machine.accept[state] is not None, expected)

    def test_priority_on_equal_length(self):
        rules = (Rule("FIRST", "a"), Rule("SECOND", "a"))
        _, raw, minimal = build(rules)
        for machine in (raw, minimal):
            self.assertEqual(tokenize("a", machine, rules), [("FIRST", "a")])

    def test_full_ascii_rule_still_has_explicit_trap(self):
        _, _, minimal = build((Rule("ANY", "[\x00-\x7f]+"),))
        self.assertEqual(minimal.transitions[minimal.trap], [minimal.trap] * 128)
        self.assertEqual(tokenize("a\0", minimal, (Rule("ANY", "[\x00-\x7f]+"),)), [("ANY", "a\0")])

    def test_rejects_empty_tokens_and_bad_regexes(self):
        for pattern in ("", "a*", "a?", "a|", "(a*)*"):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, "empty string"):
                    build_nfa((Rule("EMPTY", pattern),))
        for pattern in ("(", "a)", "[", "[]", "[z-a]", "\\", "a**", "[a", "я"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    build_nfa((Rule("BAD", pattern),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
