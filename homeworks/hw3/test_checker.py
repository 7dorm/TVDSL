"""HW3 semantic regression tests; expected types and errors are explicit."""

import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from checker import ARRAY, BOOL, ERROR, INT, check, unify


def walk(value):
    if isinstance(value, dict):
        if "type" in value:
            yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


class SemanticTests(unittest.TestCase):
    def good(self, text):
        result = check(text)
        self.assertEqual(result["diagnostics"], [], result["diagnostics"])
        self.assertIsNotNone(result["ast"])
        return result

    def bad(self, text, *codes):
        result = check(text)
        actual = [item.get("code") for item in result["diagnostics"]]
        self.assertEqual(actual, list(codes), result["diagnostics"])
        for diagnostic in result["diagnostics"]:
            self.assertEqual(diagnostic["phase"], "semantic")
            self.assertTrue(diagnostic["message"])
            self.assertIn("node_type", diagnostic)
            self.assertGreaterEqual(diagnostic["line"], 1)
            self.assertGreaterEqual(diagnostic["column"], 1)
            self.assertGreaterEqual(diagnostic["offset"], 0)
        return result

    def test_basic_function_and_complete_annotations(self):
        result = self.good("function f(x:int) returns r:int {r=x+1*2;}")
        expression_kinds = {"Name", "Integer", "Binary", "Unary", "Call", "Index", "ArraySet", "Boolean", "Quantifier"}
        for node in walk(result["ast"]):
            if node["type"] in expression_kinds:
                self.assertIn("inferred_type", node)
                self.assertNotEqual(node["inferred_type"], ERROR)
        assignment = result["ast"]["definitions"][0]["body"]["statements"][0]
        self.assertEqual(assignment["value"]["inferred_type"], INT)
        self.assertEqual(assignment["target"]["symbol_id"], result["ast"]["definitions"][0]["returns"][0]["symbol_id"])

    def test_unification_of_closed_types_and_error_propagation(self):
        for typ in (INT, ARRAY, BOOL, (INT, ARRAY)):
            self.assertTrue(unify(typ, typ))
        for left, right in ((INT, ARRAY), (INT, BOOL), (ARRAY, BOOL), ((INT, ARRAY), (ARRAY, INT))):
            self.assertFalse(unify(left, right))
        self.assertTrue(unify(INT, ERROR))

    def test_explicit_locals_arrays_conditions_and_loop(self):
        self.good("f(a:int[],n:int) requires n>=0 returns r:int ensures r>=0 uses b:int[],i:int {b=a;i=0;r=0; while(i<n) invariant i>=0 {if(b[i]>0) r=r+b[i]; else r=r+1;i=i+1;} assume r>=0;assert length(b)>=0;}")

    def test_infer_int_and_array_from_first_assignment(self):
        result = self.good("f(a:int[]) returns r:int uses x,b {x=1;b=a;b[0]=x;r=length(b);}")
        declarations = result["ast"]["definitions"][0]["locals"]
        self.assertEqual([item["annotation"] for item in declarations], [None, None])
        self.assertEqual([item["inferred_type"] for item in declarations], [INT, ARRAY])
        locals_ = [item for item in result["symbols"] if item["kind"] == "local"]
        self.assertEqual([item["inferred_type"] for item in locals_], [INT, ARRAY])

    def test_invalid_assignment_does_not_set_local_type(self):
        result = self.bad("f() returns r:int uses x {x=unknown;x=2;r=x;}", "unknown_name")
        self.assertEqual(result["ast"]["definitions"][0]["locals"][0]["inferred_type"], INT)

    def test_bad_call_does_not_infer_from_its_declared_result(self):
        result = self.bad("g(x:int) returns a:int[] uses b:int[] {a=b;} f() returns r:int uses x {x=g();x=1;r=x;}", "argument_count")
        self.assertEqual(result["ast"]["definitions"][1]["locals"][0]["inferred_type"], INT)

    def test_conflicting_local_assignments(self):
        self.bad("f(a:int[]) returns r:int uses x {x=1;x=a;r=0;}", "type_mismatch")

    def test_no_inference_without_valid_assignment(self):
        self.bad("f() returns r:int uses x {r=0;}", "cannot_infer")

    def test_inference_across_blocks_and_conflicting_branches(self):
        self.good("f() returns r:int uses x {{x=1;}r=x;}")
        self.bad("f(a:int[]) returns r:int uses x {if(true) x=1;else x=a;r=0;}", "type_mismatch")

    def test_untyped_read_before_first_assignment(self):
        self.bad("f() returns r:int uses x {r=x;x=2;}", "unknown_type")

    def test_self_reference_does_not_infer(self):
        self.bad("f() returns r:int uses x {x=x+1;r=0;}", "unknown_type")

    def test_boolean_cannot_be_inferred_as_variable_type(self):
        self.bad("f() returns r:int uses x {x=true;r=0;}", "variable_type")

    def test_unknown_variable_and_function(self):
        self.bad("f() returns r:int {r=x; r=missing(1);}", "unknown_name", "unknown_function")

    def test_assignment_does_not_declare_variable(self):
        self.bad("f() returns r:int {x=1;r=0;}", "unknown_name")

    def test_duplicate_parameter_return_and_local(self):
        self.bad("f(x:int,x:int) returns x:int uses x:int {x=1;}",
                 "duplicate_name", "duplicate_name", "duplicate_name", "immutable_parameter")

    def test_duplicate_function_and_formula_namespace(self):
        self.bad("f() returns r:int {r=0;} f()=>true;", "duplicate_name")

    def test_variables_are_separate_between_functions(self):
        self.bad("f() returns r:int uses x:int {x=1;r=x;} g() returns r:int {r=x;}", "unknown_name")

    def test_local_shadows_global_callable(self):
        self.bad("g() returns r:int {r=0;} f() returns r:int uses g:int {g=1;r=g();}", "not_callable")

    def test_contract_visibility(self):
        self.bad("f(x:int) requires r>0 returns r:int ensures y>0 uses y:int {y=0;r=x;}", "unknown_name", "unknown_name")
        self.good("f(x:int) requires x>=0 returns r:int ensures r>=x uses y:int {y=x;r=y;}")

    def test_quantifier_scope_and_annotation(self):
        result = self.good("f(a:int[]) returns r:int {assert forall(i:int|i>=0 -> a[i]>=0); assert exists(i:int|i==1); r=0;}")
        quantifiers = [node for node in walk(result["ast"]) if node["type"] == "Quantifier"]
        self.assertEqual([node["inferred_type"] for node in quantifiers], [BOOL, BOOL])
        self.assertNotEqual(quantifiers[0]["variable"]["symbol_id"], quantifiers[1]["variable"]["symbol_id"])
        self.bad("f() returns r:int {assert forall(i:int|i>=0);r=i;}", "unknown_name")

    def test_quantifier_cannot_shadow_visible_variable(self):
        self.bad("f(i:int) returns r:int {assert forall(i:int|i>=0);r=0;}", "duplicate_name")

    def test_nested_quantifier_scopes(self):
        self.good("p(a:int[])=>forall(i:int|exists(j:int|i==j)); f(a:int[]) returns r:int {assert p(a);r=0;}")
        self.bad("p()=>forall(i:int|exists(i:int|i==0));", "duplicate_name")

    def test_forward_calls_and_mutual_recursion(self):
        self.good("f(n:int) returns r:int {if(n==0) r=0;else r=g(n-1);} g(n:int) returns r:int {if(n==0) r=1;else r=f(n-1);}")

    def test_recursive_argument_count(self):
        self.bad("f(n:int) returns r:int {r=f(n,1);}", "argument_count")

    def test_wrong_argument_type_and_count(self):
        self.bad("f(x:int) returns r:int {r=x;} g(a:int[]) returns r:int {r=f(a);r=f();}", "type_mismatch", "argument_count")

    def test_formula_forward_reference_and_signature(self):
        self.good("f(a:int[]) returns r:int {assert positive(a,0);r=0;} positive(a:int[],i:int)=>a[i]>0;")
        self.bad("p(x:int)=>x>0; f() returns r:int {assert p();}", "argument_count")

    def test_formula_and_quantifier_are_predicate_only(self):
        self.bad("p()=>true; f() returns r:int {if(p()) r=1;else r=0;r=forall(i:int|i==0);}", "predicate_only", "predicate_only")

    def test_predicates_cannot_be_arithmetic_operands_or_arguments(self):
        self.bad("p()=>true; id(x:int) returns r:int {r=x;} f() returns r:int {assert -p()==0; assert id(p())==0;}",
                 "predicate_only", "predicate_only")

    def test_formula_body_and_quantifier_body_must_be_boolean(self):
        self.bad("p(x:int)=>x; q()=>forall(i:int|i);", "type_mismatch", "type_mismatch")

    def test_arithmetic_comparisons_and_boolean_rules(self):
        self.bad("f(a:int[]) returns r:int {r=true+1;assert a==a;assert not 1;assert true and 1;}",
                 "type_mismatch", "type_mismatch", "type_mismatch", "type_mismatch", "type_mismatch")
        self.good("f(x:int) returns r:int {assert not x<0 and true or false -> x>=0; r=-x;}")

    def test_all_boolean_contexts(self):
        self.bad("f() requires 1 returns r:int ensures 1 {if(1) r=0;while(1) invariant 1 {r=0;}assert 1;assume 1;}",
                 *("type_mismatch" for _ in range(7)))

    def test_assignment_type_mismatch_has_expected_actual(self):
        result = self.bad("f(a:int[]) returns r:int {r=a;}", "type_mismatch")
        self.assertEqual(result["diagnostics"][0]["expected"], INT)
        self.assertEqual(result["diagnostics"][0]["actual"], ARRAY)

    def test_parameter_and_array_parameter_are_immutable(self):
        self.bad("f(x:int,a:int[]) returns r:int {x=1;a[0]=1;r=0;}", "immutable_parameter", "immutable_parameter")

    def test_array_copy_and_update_annotation(self):
        result = self.good("f(a:int[]) returns b:int[] {b=a;b[0]=a[1]+1;}")
        update = result["ast"]["definitions"][0]["body"]["statements"][1]["value"]
        self.assertEqual((update["type"], update["inferred_type"]), ("ArraySet", ARRAY))
        self.assertEqual(update["indices"][0]["inferred_type"], INT)
        self.assertEqual(update["value"]["inferred_type"], INT)

    def test_first_array_update_constrains_local_type(self):
        result = self.good("f() returns r:int uses a {a[0]=42;r=length(a);}")
        self.assertEqual(result["ast"]["definitions"][0]["locals"][0]["inferred_type"], ARRAY)
        # An invalid update must not prevent a later valid int assignment.
        result = self.bad("f() returns r:int uses a {a[true]=1;a=2;r=a;}", "type_mismatch")
        self.assertEqual(result["ast"]["definitions"][0]["locals"][0]["inferred_type"], INT)

    def test_lowered_array_update_does_not_duplicate_unknown_name(self):
        self.bad("f() returns r:int {missing[0]=1;r=0;}", "unknown_name")

    def test_index_and_update_type_errors(self):
        self.bad("f(a:int[],x:int) returns r:int {r=x[0];r=a[a];r=a[0][0];}", "type_mismatch", "type_mismatch", "type_mismatch")
        self.bad("f(a:int[]) returns b:int[] {b=a;b[true]=a;b[0][1]=1;}", "type_mismatch", "type_mismatch", "array_dimensions")

    def test_length_builtin(self):
        self.good("f(a:int[]) returns r:int {r=length(a);}")
        self.bad("f(x:int) returns r:int {r=length(x);r=length();}", "type_mismatch", "argument_count")

    def test_tuple_return_types_and_local_inference(self):
        result = self.good("pair(a:int[]) returns n:int,b:int[] {n=length(a);b=a;} f(a:int[]) returns r:int uses x,y {x,y=pair(a);r=x+length(y);}")
        function = result["ast"]["definitions"][1]
        self.assertEqual([item["inferred_type"] for item in function["locals"]], [INT, ARRAY])
        self.assertEqual(function["body"]["statements"][0]["value"]["inferred_type"], "tuple[int, int[]]")

    def test_tuple_length_and_type_mismatch(self):
        self.bad("pair() returns x:int,y:int {x=0;y=1;} f() returns r:int uses x:int,y:int,z:int {x,y,z=pair();r=0;}", "tuple_count")
        self.bad("pair() returns x:int,y:int {x=0;y=1;} f() returns r:int uses x:int[],y:int {x,y=pair();r=0;}", "type_mismatch")

    def test_single_result_in_tuple_and_readonly_tuple_target(self):
        self.bad("one() returns r:int {r=1;} f() returns r:int uses a:int,b:int {a,b=one();r=0;}", "tuple_count")
        self.bad("pair() returns x:int,y:int {x=0;y=1;} f(x:int) returns r:int {x,r=pair();}", "immutable_parameter")

    def test_multi_return_cannot_be_used_as_scalar_or_argument(self):
        self.bad("pair() returns x:int,y:int {x=0;y=1;} f() returns r:int {r=pair()+1;}", "result_count")
        self.bad("pair() returns x:int,y:int {x=0;y=1;} id(x:int) returns r:int {r=x;} f() returns r:int {r=id(pair());}", "result_count")

    def test_missing_tuple_targets_still_report_independently(self):
        self.bad("pair() returns x:int,y:int {x=0;y=1;} f() returns r:int {x,y=pair();r=0;}", "unknown_name", "unknown_name")

    def test_several_independent_errors_and_no_cascade(self):
        self.bad("f(x:int) returns r:int {r=missing+1;x=0;assert 2;}", "unknown_name", "immutable_parameter", "type_mismatch")

    def test_exact_error_position_with_crlf(self):
        result = self.bad("f() returns r:int {\r\n r=missing;\r\n}", "unknown_name")
        diagnostic = result["diagnostics"][0]
        self.assertEqual((diagnostic["line"], diagnostic["column"], diagnostic["offset"]), (2, 4, 24))

    def test_lexical_and_syntax_errors_skip_semantics(self):
        for text, phase in (("f() returns r:int {r=@;}", "lexical"), ("f() returns r:int {r=;x=2;}", "syntax")):
            with self.subTest(phase=phase):
                result = check(text)
                self.assertEqual(result["symbols"], [])
                self.assertEqual(result["diagnostics"][0]["phase"], phase)

    def test_many_damaged_sources_terminate(self):
        rng = random.Random(3)
        tokens = ["f", "(", ")", "{", "}", "x", "=", ";", "returns", "r", ":", "int", "1", "true", "uses"]
        for _ in range(150):
            result = check(" ".join(rng.choices(tokens, k=rng.randrange(1, 35))))
            self.assertIn("diagnostics", result)

    def test_existing_ascii_course_examples_without_rewriting(self):
        root = Path(__file__).resolve().parents[2] / "project" / "examples"
        cases = {
            "step01/good_basic.funny": [],
            "step02/good_nested_calls.funny": [],
            "step04/good_nested_if.funny": [],
            "step05/good_while_basic.funny": ["unknown_name", "unknown_name", "unknown_name", "unknown_name", "unknown_name"],
            "step05/good_while_nested.funny": ["immutable_parameter"],
        }
        for file, codes in cases.items():
            with self.subTest(file=file):
                text = (root / file).read_text(encoding="utf-8")
                self.assertTrue(text.isascii())
                if codes:
                    self.bad(text, *codes)
                else:
                    self.good(text)

    def test_homework_example_files_have_expected_results(self):
        expected = {
            "good_types.funny": [],
            "good_inference.funny": [],
            "good_formulas.funny": [],
            "good_recursion.funny": [],
            "bad_names.funny": ["duplicate_name", "unknown_name"],
            "bad_types.funny": ["type_mismatch"] * 3,
            "bad_calls.funny": ["type_mismatch", "argument_count", "unknown_function"],
            "bad_parameters.funny": ["immutable_parameter"] * 2,
            "bad_inference.funny": ["type_mismatch", "cannot_infer"],
            "bad_contract_scope.funny": ["unknown_name"] * 2,
            "bad_tuple.funny": ["tuple_count", "result_count"],
        }
        folder = Path(__file__).with_name("examples")
        self.assertEqual({file.name for file in folder.glob("*.funny")}, set(expected))
        for name, codes in expected.items():
            with self.subTest(file=name):
                text = (folder / name).read_text(encoding="utf-8")
                if codes:
                    self.bad(text, *codes)
                else:
                    self.good(text)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("checker.py")), *args],
                              text=True, capture_output=True, timeout=15, check=False)

    def test_json_and_exit_codes(self):
        for text, code in (("f() returns r:int {r=1;}", 0), ("f() returns r:int {r=true;}", 1), ("", 1)):
            with self.subTest(text=text):
                result = self.run_cli("--text", text)
                self.assertEqual(result.returncode, code, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(bool(payload["diagnostics"]), bool(code))
                self.assertIn("symbols", payload)

    def test_file_and_argument_errors(self):
        file = Path(__file__).resolve().parents[2] / "project/examples/step01/good_basic.funny"
        self.assertEqual(self.run_cli("--file", str(file)).returncode, 0)
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(self.run_cli("--text", "", "--file", str(file)).returncode, 2)
        self.assertEqual(self.run_cli("--file", str(file / "missing")).returncode, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
