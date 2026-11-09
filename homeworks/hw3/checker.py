"""HW3: name resolution, type checking, local inference and annotated Funny AST."""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

from parser import parse


INT = "int"
ARRAY = "int[]"
BOOL = "bool"
ERROR = "<error>"
VARIABLE_TYPES = {INT, ARRAY}


def type_name(typ):
    if isinstance(typ, tuple):
        return "tuple[" + ", ".join(typ) + "]"
    return typ if typ is not None else "<unknown>"


def unify(expected, actual):
    return ERROR in (expected, actual) or expected == actual


@dataclass
class Symbol:
    id: int
    name: str
    kind: str
    scope: str
    pos: dict
    typ: str | None
    mutable: bool = False
    parameters: tuple[str, ...] = ()
    returns: tuple[str, ...] = ()
    declarations: list = field(default_factory=list)

    def dump(self):
        result = dict(id=self.id, name=self.name, kind=self.kind, scope=self.scope,
                      pos=self.pos, inferred_type=type_name(self.typ), mutable=self.mutable)
        if self.kind in {"function", "formula", "builtin"}:
            result.update(parameters=list(self.parameters), returns=list(self.returns))
        return result


class Scope:
    def __init__(self, label, parent=None):
        self.label, self.parent, self.symbols = label, parent, {}

    def lookup(self, name):
        if name in self.symbols:
            return self.symbols[name]
        return self.parent.lookup(name) if self.parent is not None else None


class SemanticAnalyzer:
    def __init__(self):
        self.globals = {}
        self.symbols = []
        self.diagnostics = []
        self.reported = set()
        self.unresolved_reported = set()
        self.scope_number = 0

    def report(self, node, code, message, expected=None, actual=None):
        key = (code, node["pos"]["offset"], message)
        if key in self.reported:
            return
        self.reported.add(key)
        diagnostic = dict(phase="semantic", code=code, message=message,
                          node_type=node["type"], **node["pos"])
        if expected is not None:
            diagnostic["expected"] = type_name(expected)
        if actual is not None:
            diagnostic["actual"] = type_name(actual)
        self.diagnostics.append(diagnostic)

    def new_symbol(self, name, kind, scope, pos, typ, **fields):
        symbol = Symbol(len(self.symbols), name, kind, scope, pos, typ, **fields)
        self.symbols.append(symbol)
        return symbol

    def annotate(self, node, typ, symbol=None):
        node["inferred_type"] = type_name(typ)
        if symbol is not None:
            node["symbol_id"] = symbol.id
        return typ

    def require(self, node, actual, expected, context):
        if not unify(expected, actual):
            self.report(node, "type_mismatch",
                        f"{context}: ожидался {type_name(expected)}, получен {type_name(actual)}",
                        expected, actual)
            return False
        return actual != ERROR

    def declare(self, node, scope, kind):
        name = node["name"]
        # Function declarations share a single namespace; quantifiers may not
        # shadow any variable visible in the current predicate.
        previous = scope.lookup(name) if kind == "quantifier" else scope.symbols.get(name)
        if previous is not None:
            self.report(node, "duplicate_name", f"Повторное объявление имени '{name}'")
            self.annotate(node, ERROR)
            return previous
        symbol = self.new_symbol(name, kind, scope.label, node["pos"], node["annotation"],
                                 mutable=kind in {"local", "return"})
        symbol.declarations.append(node)
        scope.symbols[name] = symbol
        self.annotate(node, symbol.typ, symbol)
        return symbol

    def register_definitions(self, ast):
        builtin = self.new_symbol("length", "builtin", "module", dict(offset=0, line=1, column=1),
                                  "func(int[]) -> (int)", parameters=(ARRAY,), returns=(INT,))
        self.globals[builtin.name] = builtin
        for definition in ast["definitions"]:
            kind = "function" if definition["type"] == "Function" else "formula"
            params = tuple(item["annotation"] for item in definition["params"])
            results = tuple(item["annotation"] for item in definition["returns"]) if kind == "function" else (BOOL,)
            signature = f"{kind}({', '.join(params)}) -> ({', '.join(results)})"
            name = definition["name"]
            if name in self.globals:
                self.report(definition, "duplicate_name", f"Повторное определение '{name}'")
                self.annotate(definition, ERROR)
                continue
            symbol = self.new_symbol(name, kind, "module", definition["pos"], signature,
                                     parameters=params, returns=results)
            self.globals[name] = symbol
            self.annotate(definition, signature, symbol)

    def definition(self, definition):
        self.scope_number += 1
        label = f"{definition['name']}#{self.scope_number}"
        base = Scope(f"{label}:parameters")
        for declaration in definition["params"]:
            self.declare(declaration, base, "parameter")
        if definition["type"] == "Formula":
            self.predicate(definition["predicate"], base, "Тело формулы")
            return
        # requires only sees parameters. ensures additionally sees results.
        if definition["requires"] is not None:
            self.predicate(definition["requires"], base, "requires")
        base.label = f"{label}:results"
        for declaration in definition["returns"]:
            self.declare(declaration, base, "return")
        if definition["ensures"] is not None:
            self.predicate(definition["ensures"], base, "ensures")
        base.label = f"{label}:locals"
        locals_ = [self.declare(declaration, base, "local") for declaration in definition["locals"]]
        self.statement(definition["body"], base)
        for symbol in locals_:
            if symbol.typ is None and symbol.id not in self.unresolved_reported:
                self.report(symbol.declarations[0], "cannot_infer",
                            f"Невозможно вывести тип локальной переменной '{symbol.name}': нет корректного присваивания")
                self.unresolved_reported.add(symbol.id)

    def resolve_variable(self, node, scope, reading=True):
        symbol = scope.lookup(node["name"])
        if symbol is None:
            self.report(node, "unknown_name", f"Переменная '{node['name']}' не объявлена в этой области")
            self.annotate(node, ERROR)
            return None
        if reading and symbol.typ is None:
            self.report(node, "unknown_type",
                        f"Тип '{symbol.name}' ещё не выведен: нужно предыдущее корректное присваивание")
            self.unresolved_reported.add(symbol.id)
        self.annotate(node, symbol.typ if symbol.typ is not None else ERROR, symbol)
        return symbol

    def predicate(self, node, scope, context):
        typ = self.expression(node, scope, predicate=True)
        return self.require(node, typ, BOOL, context)

    def call(self, node, scope, predicate, multiple):
        argument_types = [self.expression(arg, scope, predicate=False) for arg in node["args"]]
        shadow = scope.lookup(node["name"])
        if shadow is not None:
            self.report(node, "not_callable", f"'{node['name']}' — переменная, её нельзя вызвать")
            return self.annotate(node, ERROR, shadow)
        signature = self.globals.get(node["name"])
        if signature is None:
            self.report(node, "unknown_function", f"Функция или формула '{node['name']}' не определена")
            return self.annotate(node, ERROR)
        valid = True
        if len(argument_types) != len(signature.parameters):
            self.report(node, "argument_count",
                        f"Вызов '{signature.name}': ожидалось аргументов {len(signature.parameters)}, получено {len(argument_types)}")
            valid = False
        for index, (argument, actual, expected) in enumerate(zip(node["args"], argument_types, signature.parameters), 1):
            valid = self.require(argument, actual, expected, f"Аргумент {index} вызова '{signature.name}'") and valid
        if signature.kind == "formula" and (not predicate or multiple):
            self.report(node, "predicate_only", f"Формула '{signature.name}' разрешена только в предикатах")
            valid = False
        if not multiple and len(signature.returns) != 1:
            self.report(node, "result_count",
                        f"Функция '{signature.name}' возвращает {len(signature.returns)} значений: требуется присваивание кортежа")
            valid = False
        result = signature.returns if multiple else signature.returns[0]
        return self.annotate(node, result if valid else ERROR, signature)

    def expression(self, node, scope, predicate=False, multiple=False):
        kind = node["type"]
        if kind == "Integer":
            return self.annotate(node, INT)
        if kind == "Boolean":
            return self.annotate(node, BOOL)
        if kind == "Name":
            symbol = self.resolve_variable(node, scope)
            return symbol.typ if symbol is not None and symbol.typ is not None else ERROR
        if kind == "Call":
            return self.call(node, scope, predicate, multiple)
        if kind == "Unary":
            expected = INT if node["operator"] == "-" else BOOL
            actual = self.expression(node["operand"], scope, predicate and node["operator"] == "not")
            valid = self.require(node["operand"], actual, expected, f"Оператор '{node['operator']}'")
            return self.annotate(node, expected if valid else ERROR)
        if kind == "Binary":
            operator = node["operator"]
            arithmetic = operator in {"+", "-", "*", "/"}
            comparison = operator in {"==", "!=", "<", "<=", ">", ">="}
            expected = INT if arithmetic or comparison else BOOL
            # Formula references/quantifiers are only allowed in predicate
            # positions, never inside arithmetic operands or call arguments.
            child_predicate = predicate and not (arithmetic or comparison)
            left = self.expression(node["left"], scope, child_predicate)
            right = self.expression(node["right"], scope, child_predicate)
            left_ok = self.require(node["left"], left, expected, f"Левый операнд '{operator}'")
            right_ok = self.require(node["right"], right, expected, f"Правый операнд '{operator}'")
            result = INT if arithmetic else BOOL
            return self.annotate(node, result if left_ok and right_ok else ERROR)
        if kind == "Index":
            array = self.expression(node["array"], scope, predicate=False)
            index = self.expression(node["index"], scope, predicate=False)
            array_ok = self.require(node["array"], array, ARRAY, "Индексируемое значение")
            index_ok = self.require(node["index"], index, INT, "Индекс массива")
            return self.annotate(node, INT if array_ok and index_ok else ERROR)
        if kind == "ArraySet":
            # a[i] = value constrains an untyped local a to int[]. Bind it in
            # assign() only if this entire update has valid indices and value.
            symbol = scope.lookup(node["array"]["name"])
            if symbol is not None and symbol.typ is None and symbol.mutable:
                array = self.annotate(node["array"], ARRAY, symbol)
            else:
                array = self.expression(node["array"], scope, predicate=False)
            valid = self.require(node["array"], array, ARRAY, "Обновляемый массив")
            for index in node["indices"]:
                actual = self.expression(index, scope, predicate=False)
                valid = self.require(index, actual, INT, "Индекс массива") and valid
            value = self.expression(node["value"], scope, predicate=False)
            valid = self.require(node["value"], value, INT, "Новое значение элемента массива") and valid
            if len(node["indices"]) != 1:
                self.report(node, "array_dimensions", "Тип int[] поддерживает только один индекс")
                valid = False
            return self.annotate(node, ARRAY if valid else ERROR)
        if kind == "Quantifier":
            if not predicate:
                self.report(node, "predicate_only", "Квантор разрешён только в предикатах")
            self.scope_number += 1
            child = Scope(f"{scope.label}:quantifier#{self.scope_number}", scope)
            before = len(self.diagnostics)
            self.declare(node["variable"], child, "quantifier")
            valid = self.predicate(node["predicate"], child, "Тело квантора")
            return self.annotate(node, BOOL if predicate and valid and len(self.diagnostics) == before else ERROR)
        raise ValueError(f"Unsupported AST expression: {kind}")

    def assign(self, target, actual, scope):
        symbol = self.resolve_variable(target, scope, reading=False)
        if symbol is None:
            return
        if not symbol.mutable:
            self.report(target, "immutable_parameter", f"Нельзя присваивать параметру '{symbol.name}'")
            return
        if actual == ERROR:
            return
        if symbol.typ is None:
            if actual not in VARIABLE_TYPES:
                self.report(target, "variable_type", f"Переменная '{symbol.name}' должна иметь тип int или int[], получен {type_name(actual)}")
                self.unresolved_reported.add(symbol.id)
                return
            symbol.typ = actual
            for declaration in symbol.declarations:
                self.annotate(declaration, actual, symbol)
        valid = self.require(target, actual, symbol.typ, f"Присваивание '{symbol.name}'")
        self.annotate(target, symbol.typ if valid else ERROR, symbol)

    def statement(self, node, scope):
        kind = node["type"]
        if kind == "Block":
            self.scope_number += 1
            child = Scope(f"{scope.label}:block#{self.scope_number}", scope)
            for statement in node["statements"]:
                self.statement(statement, child)
        elif kind == "Assign":
            actual = self.expression(node["value"], scope)
            self.assign(node["target"], actual, scope)
        elif kind == "TupleAssign":
            result = self.expression(node["value"], scope, multiple=True)
            if result != ERROR and len(result) != len(node["targets"]):
                self.report(node, "tuple_count",
                            f"Присваивание кортежа: переменных {len(node['targets'])}, результатов {len(result)}")
                result = ERROR
            for index, target in enumerate(node["targets"]):
                self.assign(target, result[index] if result != ERROR else ERROR, scope)
        elif kind == "If":
            actual = self.expression(node["condition"], scope)
            self.require(node["condition"], actual, BOOL, "Условие if")
            self.statement(node["then"], scope)
            if node["otherwise"] is not None:
                self.statement(node["otherwise"], scope)
        elif kind == "While":
            actual = self.expression(node["condition"], scope)
            self.require(node["condition"], actual, BOOL, "Условие while")
            if node["invariant"] is not None:
                self.predicate(node["invariant"], scope, "invariant")
            self.statement(node["body"], scope)
        elif kind in {"Assert", "Assume"}:
            self.predicate(node["predicate"], scope, kind.lower())
        else:
            raise ValueError(f"Unsupported AST statement: {kind}")

    def run(self, ast):
        self.register_definitions(ast)
        for definition in ast["definitions"]:
            self.definition(definition)
        return dict(ast=ast, symbols=[symbol.dump() for symbol in self.symbols],
                    diagnostics=self.diagnostics)


def check(text):
    result = parse(text)
    if result["diagnostics"]:
        return dict(ast=result["ast"], symbols=[], diagnostics=result["diagnostics"])
    analyzer = SemanticAnalyzer()
    try:
        return analyzer.run(result["ast"])
    except RecursionError:
        analyzer.report(result["ast"], "nesting", "Слишком большая вложенность для семантического анализа")
        return dict(ast=None, symbols=[], diagnostics=analyzer.diagnostics)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Funny source text")
    source.add_argument("--file", type=Path, help="UTF-8 Funny source file")
    args = cli.parse_args()
    try:
        text = args.text if args.text is not None else args.file.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        cli.exit(1, f"{error}\n")
    result = check(text)
    try:
        output = json.dumps(result, ensure_ascii=False, indent=2)
    except RecursionError:
        output = json.dumps(dict(ast=None, symbols=[], diagnostics=[dict(
            phase="semantic", code="nesting", message="Слишком глубокий AST для вывода JSON",
            node_type="Module", offset=0, line=1, column=1)]), ensure_ascii=False, indent=2)
        print(output)
        raise SystemExit(1)
    print(output)
    raise SystemExit(1 if result["diagnostics"] else 0)


if __name__ == "__main__":
    main()
