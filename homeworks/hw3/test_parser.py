"""HW2 tests. Run: python -B -m unittest -v test_parser.py."""

import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

from lexer import LexError, tokenize
from parser import parse


def shape(value):
    if isinstance(value, dict):
        return {key: shape(child) for key, child in value.items() if key != "pos"}
    if isinstance(value, list):
        return [shape(child) for child in value]
    return value


def name(text):
    return dict(type="Name", name=text)


def integer(text):
    return dict(type="Integer", value=str(text))


def binary(operator, left, right):
    return dict(type="Binary", operator=operator, left=left, right=right)


POSITIVE = [
    ("minimal", "f() returns r:int {}"),
    ("prefix", "function f(x:int) returns r:int {r=x+1;}"),
    ("contracts", "f(x:int) requires x>=0 returns r:int ensures r>x uses y, a:int[] {r=x+1;}"),
    ("empty_contracts", "f() requires returns r:int ensures uses x {while(true) invariant {r=1;}}"),
    ("single_body", "f() returns r:int r=1;"),
    ("multiple_returns", "f(a:int,b:int) returns x:int,y:int {x=b;y=a;}"),
    ("tuple", "f() returns r:int {x,y=swap(1,2);r=x-y;}"),
    ("arrays", "f(a:int[]) returns b:int[] {b=a;b[0]=a[1]+length(a);b[0][1]=2;}"),
    ("if_else", "f(x:int) returns r:int {if(x<0) r=1; else r=2;}"),
    ("loop", "f(x:int) returns r:int {while(x>0) invariant r>=0 {r=r+1;}}"),
    ("checks", "f() returns r:int {assume true;assert not false and true or false;}"),
    ("formula", "positive(x:int)=>x>0; f(x:int) requires positive(x) returns r:int {r=x;}"),
    ("quantifiers", "f(a:int[]) returns r:int {assert forall(i:int|i>=0 -> exists(j:int|a[i]==j));}"),
    ("parentheses", "f(x:int,y:int) returns r:int {if((x+1)<y and (x<y or true)) r=length(a)[0];}"),
    ("recursive_calls", "fact(n:int) returns r:int {if(n<=1) r=1;else r=n*fact(n-1);}"),
    ("comments", "// heading\r\nf() returns r:int {r=1;// tail\r\n}"),
]
NEGATIVE = [
    ("empty", ""), ("whitespace", " \t\r\n"), ("comment_only", "// comment"),
    ("missing_paren", "f(x:int returns r:int {}"),
    ("missing_brace", "f() returns r:int {r=1;"),
    ("missing_semicolon", "f() returns r:int {r=1}"),
    ("missing_parameter_type", "f(x) returns r:int {}"),
    ("missing_return_type", "f() returns r {}"),
    ("missing_return", "f() {}"),
    ("bad_operator", "f() returns r:int {r=1+;}"),
    ("unexpected_statement", "f() returns r:int {else r=1;}"),
    ("extra_token", "f() returns r:int {} ;"),
    ("wrong_bracket", "f() returns r:int {r=a[1);} "),
    ("extra_comparison", "f() returns r:int {assert a<b<c;}"),
    ("bad_tuple", "f() returns r:int {a,b=1+2;}"),
    ("bad_array_target", "f() returns r:int {a[0],b=f();}"),
    ("bad_type", "f() returns r:bool {}"),
    ("standalone_call", "f() returns r:int {g();}"),
    ("prefixed_formula", "function p(x:int)=>x>0;"),
    ("missing_quantifier_pipe", "p()=>forall(i:int i>0);"),
    ("bare_length", "f() returns r:int {r=length;}"),
    ("leading_zero", "f() returns r:int {r=01;}"),
    ("non_ascii", "f() returns r:int {// я\n}"),
    ("unknown_character", "f() returns r:int {r=@;}"),
]


class ParserTests(unittest.TestCase):
    def successful(self, text):
        result = parse(text)
        self.assertEqual(result["diagnostics"], [], text)
        return result["ast"]

    def value(self, text):
        ast = self.successful("f() returns r:int {r=" + text + ";}")
        return shape(ast["definitions"][0]["body"]["statements"][0]["value"])

    def test_positive_programs(self):
        for case, text in POSITIVE:
            with self.subTest(case=case):
                self.successful(text)

    def test_negative_programs_have_positioned_diagnostics(self):
        for case, text in NEGATIVE:
            with self.subTest(case=case):
                result = parse(text)
                self.assertTrue(result["diagnostics"], case)
                for diagnostic in result["diagnostics"]:
                    self.assertIn(diagnostic["phase"], {"lexical", "syntax"})
                    self.assertTrue(diagnostic["message"])
                    self.assertGreaterEqual(diagnostic["line"], 1)
                    self.assertGreaterEqual(diagnostic["column"], 1)
                    self.assertTrue(0 <= diagnostic["offset"] <= len(text))

    def test_exact_ast_snapshot(self):
        actual = shape(self.successful("function f(x:int) returns r:int {r=x+1;}"))
        expected = dict(type="Module", definitions=[dict(
            type="Function", name="f", params=[dict(type="VarDecl", name="x", annotation="int")],
            returns=[dict(type="VarDecl", name="r", annotation="int")], requires=None, ensures=None, locals=[],
            body=dict(type="Block", statements=[dict(type="Assign", target=name("r"),
                                                     value=binary("+", name("x"), integer(1)))]),
        )])
        self.assertEqual(actual, expected)

    def test_arithmetic_precedence_associativity_and_unary(self):
        self.assertEqual(self.value("1+2*3"), binary("+", integer(1), binary("*", integer(2), integer(3))))
        self.assertEqual(self.value("8-3-1"), binary("-", binary("-", integer(8), integer(3)), integer(1)))
        self.assertEqual(self.value("8/4/2"), binary("/", binary("/", integer(8), integer(4)), integer(2)))
        self.assertEqual(self.value("(1+2)*3"), binary("*", binary("+", integer(1), integer(2)), integer(3)))
        self.assertEqual(self.value("-x*2"), binary("*", dict(type="Unary", operator="-", operand=name("x")), integer(2)))
        self.assertEqual(self.value("--1"), dict(type="Unary", operator="-", operand=dict(type="Unary", operator="-", operand=integer(1))))

    def test_boolean_precedence_and_right_associative_implication(self):
        self.assertEqual(self.value("not x<y and a or b -> c -> d"), binary("->",
            binary("or", binary("and", dict(type="Unary", operator="not", operand=binary("<", name("x"), name("y"))), name("a")), name("b")),
            binary("->", name("c"), name("d"))))

    def test_calls_indices_and_quantifiers_snapshot(self):
        self.assertEqual(self.value("f(1,g(2))[i]"), dict(type="Index", array=dict(type="Call", name="f", args=[integer(1),
            dict(type="Call", name="g", args=[integer(2)])]), index=name("i")))
        self.assertEqual(self.value("forall(i:int|i>=0)"), dict(type="Quantifier", quantifier="forall",
            variable=dict(type="VarDecl", name="i", annotation="int"), predicate=binary(">=", name("i"), integer(0))))

    def test_array_update_desugars_without_duplicating_index_calls(self):
        ast = self.successful("f() returns a:int[] {a[i()][j()]=v();}")
        update = shape(ast["definitions"][0]["body"]["statements"][0])
        self.assertEqual(update, dict(type="Assign", target=name("a"), value=dict(
            type="ArraySet", array=name("a"), indices=[dict(type="Call", name="i", args=[]),
            dict(type="Call", name="j", args=[])], value=dict(type="Call", name="v", args=[]))))
        encoded = json.dumps(update)
        for function in ("i", "j", "v"):
            self.assertEqual(encoded.count('"name": "' + function + '"'), 1)

    def test_tuple_assignment_snapshot(self):
        ast = self.successful("f() returns r:int {a,b=swap(1,2);}")
        self.assertEqual(shape(ast["definitions"][0]["body"]["statements"][0]), dict(
            type="TupleAssign", targets=[name("a"), name("b")],
            value=dict(type="Call", name="swap", args=[integer(1), integer(2)])))

    def test_dangling_else_belongs_to_inner_if(self):
        ast = self.successful("f() returns r:int {if(a) if(b) r=1; else r=2;}")
        outer = ast["definitions"][0]["body"]["statements"][0]
        self.assertIsNone(outer["otherwise"])
        self.assertEqual(outer["then"]["otherwise"]["value"]["value"], "2")

    def test_declarations_contracts_and_formula(self):
        ast = self.successful("p(x:int)=>x>0; f(x:int) requires p(x) returns r:int ensures r>x uses y,z:int[] {r=x;}")
        formula, function = ast["definitions"]
        self.assertEqual(formula["type"], "Formula")
        self.assertEqual(formula["predicate"]["operator"], ">")
        self.assertEqual(function["requires"]["type"], "Call")
        self.assertEqual(function["ensures"]["operator"], ">")
        self.assertEqual([decl["annotation"] for decl in function["locals"]], [None, "int[]"])

    def test_ast_positions(self):
        ast = self.successful("f() returns r:int {\r\n r = 1 + 2;\r\n}")
        assignment = ast["definitions"][0]["body"]["statements"][0]
        self.assertEqual((assignment["pos"]["line"], assignment["pos"]["column"]), (2, 2))
        self.assertEqual((assignment["value"]["pos"]["line"], assignment["value"]["pos"]["column"]), (2, 8))

    def test_panic_recovery_retains_following_statements_and_definitions(self):
        text = "f() returns r:int {\n r = ;\n x=2;\n assert ;\n r=3;\n} g() returns s:int {s=4;}"
        result = parse(text)
        self.assertEqual(len(result["diagnostics"]), 2)
        self.assertEqual([(d["line"], d["column"]) for d in result["diagnostics"]], [(2, 6), (4, 9)])
        functions = result["ast"]["definitions"]
        self.assertEqual([item["name"] for item in functions], ["f", "g"])
        self.assertEqual([item["target"]["name"] for item in functions[0]["body"]["statements"]], ["x", "r"])

    def test_missing_semicolon_recovery_does_not_consume_next_statement(self):
        result = parse("f() returns r:int {\n r=1\n x=2;\n}")
        self.assertEqual([(d["line"], d["column"]) for d in result["diagnostics"]], [(3, 2)])
        statements = result["ast"]["definitions"][0]["body"]["statements"]
        self.assertEqual([statement["target"]["name"] for statement in statements], ["x"])

    def test_closing_brace_is_preserved_during_recovery(self):
        result = parse("f() returns r:int {r=} g() returns s:int {}")
        self.assertEqual(len(result["diagnostics"]), 1)
        self.assertEqual([item["name"] for item in result["ast"]["definitions"]], ["f", "g"])

    def test_bad_header_recovers_to_next_definition(self):
        result = parse("bad(x) returns r:int {r=1;} good() returns r:int {r=2;}")
        self.assertEqual(len(result["diagnostics"]), 1)
        self.assertEqual([item["name"] for item in result["ast"]["definitions"]], ["good"])

    def test_semantic_errors_are_deferred(self):
        self.successful("f(x:int) returns r:int {if(42) x=unknown; r=true+1;}")

    def test_large_integer_stays_a_lexeme(self):
        digits = "9" * 5000
        self.assertEqual(self.value(digits), integer(digits))

    def test_excessive_nesting_reports_diagnostic(self):
        text = "f() returns r:int {r=" + "(" * 400 + "1" + ")" * 400 + ";}"
        result = parse(text)
        self.assertTrue(result["diagnostics"])
        self.assertIn("Nesting", result["diagnostics"][-1]["message"])

    def test_damaged_inputs_terminate(self):
        rng = random.Random(2026)
        fragments = ["f", "(", ")", "{", "}", "[", "]", ";", "=", "1", "if", "else", "returns", "int", ":", ",", "assert", "not"]
        for case in range(500):
            text = " ".join(rng.choices(fragments, k=rng.randrange(1, 45)))
            with self.subTest(case=case):
                result = parse(text)
                self.assertLessEqual(len(result["diagnostics"]), len(tokenize(text)) + 1)

    def test_course_corpus_respects_lexical_syntax_and_semantic_boundaries(self):
        examples = sorted((Path(__file__).resolve().parents[2] / "project" / "examples").rglob("*.funny"))
        self.assertTrue(examples)
        syntax_bad = {"bad_missing_semicolon.funny", "bad_missing_paren.funny"}
        for file in examples:
            text = file.read_bytes().decode("utf-8")
            with self.subTest(file=file.name):
                result = parse(text)
                if not text.isascii():
                    self.assertEqual(result["diagnostics"][0]["phase"], "lexical")
                elif file.name in syntax_bad:
                    self.assertTrue(result["diagnostics"])
                else:
                    self.assertEqual(result["diagnostics"], [])


class LexerTests(unittest.TestCase):
    def test_positions_cr_lf_crlf_tabs_comments_and_eof(self):
        tokens = tokenize("a\r\n//x\r\n\tb\nc\rd")
        self.assertEqual([(t.lexeme, t.offset, t.line, t.column) for t in tokens], [
            ("a", 0, 1, 1), ("b", 9, 3, 2), ("c", 11, 4, 1), ("d", 13, 5, 1), ("", 14, 5, 2),
        ])

    def test_lexer_errors_have_exact_locations(self):
        for text, line, column, offset in [("\r\n@", 2, 1, 2), ("x\n01", 2, 1, 2), ("// я", 1, 4, 3)]:
            with self.subTest(input=text):
                with self.assertRaises(LexError) as caught:
                    tokenize(text)
                self.assertEqual(caught.exception.diagnostic, dict(phase="lexical",
                    message=caught.exception.diagnostic["message"], line=line, column=column, offset=offset))

    def test_every_ascii_code_in_comment_and_rejected_inputs(self):
        for code in range(128):
            with self.subTest(code=code):
                self.assertEqual([token.kind for token in tokenize("//" + chr(code))], ["EOF"])
        for text in ("00", "01", "\0", "\v", "\f", "\x7f", "я", "→", "// я", "// 😀"):
            with self.subTest(input=text):
                with self.assertRaises(LexError):
                    tokenize(text)

    def test_token_priority_and_longest_match(self):
        tokens = tokenize("if ifx int 0 == = // skip\n_a")
        self.assertEqual([(t.kind, t.lexeme) for t in tokens], [
            ("IF", "if"), ("IDENT", "ifx"), ("TYPE_INT", "int"), ("INT", "0"),
            ("EQ", "=="), ("ASSIGN", "="), ("IDENT", "_a"), ("EOF", ""),
        ])


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("parser.py")), *args],
                              text=True, capture_output=True, check=False, timeout=15)

    def test_success_and_error_json_with_exit_codes(self):
        for text, code in [("f() returns r:int {r=1;}", 0), ("", 1), ("f() returns r:int {r=;}", 1)]:
            result = self.run_cli("--text", text)
            self.assertEqual(result.returncode, code, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(bool(payload["diagnostics"]), code == 1)

    def test_file_input_and_invalid_arguments(self):
        file = Path(__file__).resolve().parents[2] / "project" / "examples" / "step01" / "good_basic.funny"
        result = self.run_cli("--file", str(file))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["diagnostics"], [])
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(self.run_cli("--text", "", "--file", str(file)).returncode, 2)
        self.assertEqual(self.run_cli("--file", str(file / "missing.funny")).returncode, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
