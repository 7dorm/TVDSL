"""HW2: predictive recursive descent for Funny; JSON AST and panic recovery."""

import argparse
import json
from pathlib import Path

from lexer import LexError, tokenize

COMPARISONS = {"EQ", "NE", "LT", "LE", "GT", "GE"}
EXPR_START = {"IDENT", "LENGTH", "INT", "TRUE", "FALSE", "LPAREN", "MINUS", "NOT", "FORALL", "EXISTS"}
STMT_START = {"IDENT", "LBRACE", "IF", "WHILE", "ASSERT", "ASSUME"}


def node(kind, token, **fields):
    return dict(type=kind, pos=token.position(), **fields)


class ParseError(ValueError):
    def __init__(self, token, message):
        self.diagnostic = dict(phase="syntax", message=message, **token.position())
        super().__init__(f"{token.line}:{token.column}: {message}")


class Parser:
    def __init__(self, tokens):
        self.tokens, self.index, self.diagnostics = tokens, 0, []

    @property
    def current(self):
        return self.tokens[self.index]

    def advance(self):
        token = self.current
        if token.kind != "EOF":
            self.index += 1
        return token

    def take(self, kind):
        return self.advance() if self.current.kind == kind else None

    def expect(self, kind):
        token = self.take(kind)
        if token is None:
            raise ParseError(self.current, f"Expected {kind}, found {self.current.kind} {self.current.lexeme!r}")
        return token

    def report(self, error):
        self.diagnostics.append(error.diagnostic)

    def synchronize_statement(self, before):
        if self.index == before and self.current.kind not in {"EOF", "RBRACE"}:
            self.advance()
        while self.current.kind not in {"SEMICOLON", "RBRACE", "EOF"} | STMT_START:
            self.advance()
        self.take("SEMICOLON")

    def synchronize_definition(self, before):
        if self.index == before:
            self.advance()
        depth = 0
        while self.current.kind != "EOF":
            kind = self.current.kind
            next_kind = self.tokens[min(self.index + 1, len(self.tokens) - 1)].kind
            if depth == 0 and (kind == "FUNCTION" or kind == "IDENT" and next_kind == "LPAREN"):
                return
            if kind == "LBRACE":
                depth += 1
            elif kind == "RBRACE":
                depth = max(0, depth - 1)
            self.advance()

    def module(self):
        start, definitions = self.current, []
        if self.current.kind == "EOF":
            self.report(ParseError(self.current, "Expected at least one definition"))
        while self.current.kind != "EOF":
            before = self.index
            try:
                definitions.append(self.definition())
            except ParseError as error:
                self.report(error)
                self.synchronize_definition(before)
        return node("Module", start, definitions=definitions)

    def declaration(self, optional_type=False):
        name = self.expect("IDENT")
        annotation = None
        if not optional_type or self.current.kind == "COLON":
            self.expect("COLON")
            self.expect("TYPE_INT")
            annotation = "int"
            if self.take("LBRACKET"):
                self.expect("RBRACKET")
                annotation = "int[]"
        return node("VarDecl", name, name=name.lexeme, annotation=annotation)

    def declarations(self, optional_type=False):
        result = [self.declaration(optional_type)]
        while self.take("COMMA"):
            result.append(self.declaration(optional_type))
        return result

    def contract(self, keyword):
        if self.take(keyword) is None:
            return None
        return self.expression() if self.current.kind in EXPR_START else None

    def definition(self):
        prefix = self.take("FUNCTION")
        name = self.expect("IDENT")
        self.expect("LPAREN")
        params = [] if self.current.kind == "RPAREN" else self.declarations()
        self.expect("RPAREN")
        if self.take("ARROW"):
            if prefix:
                raise ParseError(name, "A formula cannot use the function prefix")
            predicate = self.expression()
            self.take("SEMICOLON")
            return node("Formula", name, name=name.lexeme, params=params, predicate=predicate)
        requires = self.contract("REQUIRES")
        self.expect("RETURNS")
        returns = self.declarations()
        ensures = self.contract("ENSURES")
        locals_ = self.declarations(True) if self.take("USES") else []
        body = self.statement()
        return node("Function", prefix or name, name=name.lexeme, params=params,
                    returns=returns, requires=requires, ensures=ensures, locals=locals_, body=body)

    def block(self):
        start, statements = self.expect("LBRACE"), []
        while self.current.kind not in {"RBRACE", "EOF"}:
            before = self.index
            try:
                statements.append(self.statement())
            except ParseError as error:
                self.report(error)
                self.synchronize_statement(before)
        self.expect("RBRACE")
        return node("Block", start, statements=statements)

    def statement(self):
        token = self.current
        if token.kind == "LBRACE":
            return self.block()
        if self.take("IF"):
            self.expect("LPAREN")
            condition = self.expression()
            self.expect("RPAREN")
            then = self.statement()
            otherwise = self.statement() if self.take("ELSE") else None
            return node("If", token, condition=condition, then=then, otherwise=otherwise)
        if self.take("WHILE"):
            self.expect("LPAREN")
            condition = self.expression()
            self.expect("RPAREN")
            invariant = self.contract("INVARIANT")
            return node("While", token, condition=condition, invariant=invariant, body=self.statement())
        if token.kind in {"ASSERT", "ASSUME"}:
            self.advance()
            predicate = self.expression()
            self.expect("SEMICOLON")
            return node("Assert" if token.kind == "ASSERT" else "Assume", token, predicate=predicate)
        if token.kind == "IDENT":
            return self.assignment()
        raise ParseError(token, f"Expected statement, found {token.kind} {token.lexeme!r}")

    def assignment(self):
        name = self.expect("IDENT")
        target = node("Name", name, name=name.lexeme)
        indices = []
        while self.take("LBRACKET"):
            indices.append(self.expression())
            self.expect("RBRACKET")
        targets = [target]
        while not indices and self.take("COMMA"):
            extra = self.expect("IDENT")
            targets.append(node("Name", extra, name=extra.lexeme))
        self.expect("ASSIGN")
        value = self.expression()
        self.expect("SEMICOLON")
        if len(targets) > 1:
            if value["type"] != "Call":
                raise ParseError(name, "Tuple assignment requires a function call")
            return node("TupleAssign", name, targets=targets, value=value)
        if indices:
            value = node("ArraySet", name, array=target, indices=indices, value=value)
        return node("Assign", name, target=target, value=value)

    def chain(self, operand, kinds):
        left = operand()
        while self.current.kind in kinds:
            operator = self.advance()
            left = node("Binary", operator, operator=operator.lexeme, left=left, right=operand())
        return left

    def expression(self):
        left = self.disjunction()
        operator = self.take("IMPLIES")
        if operator:
            return node("Binary", operator, operator="->", left=left, right=self.expression())
        return left

    def disjunction(self):
        return self.chain(self.conjunction, {"OR"})

    def conjunction(self):
        return self.chain(self.negation, {"AND"})

    def negation(self):
        operator = self.take("NOT")
        if operator:
            return node("Unary", operator, operator="not", operand=self.negation())
        return self.comparison()

    def comparison(self):
        left = self.additive()
        if self.current.kind in COMPARISONS:
            operator = self.advance()
            return node("Binary", operator, operator=operator.lexeme, left=left, right=self.additive())
        return left

    def additive(self):
        return self.chain(self.multiplicative, {"PLUS", "MINUS"})

    def multiplicative(self):
        return self.chain(self.unary, {"STAR", "SLASH"})

    def unary(self):
        operator = self.take("MINUS")
        if operator:
            return node("Unary", operator, operator="-", operand=self.unary())
        return self.postfix()

    def postfix(self):
        value = self.atom()
        while self.current.kind == "LBRACKET":
            bracket = self.advance()
            index = self.expression()
            self.expect("RBRACKET")
            value = node("Index", bracket, array=value, index=index)
        return value

    def atom(self):
        token = self.current
        if token.kind not in {"INT", "TRUE", "FALSE", "IDENT", "LENGTH", "LPAREN", "FORALL", "EXISTS"}:
            raise ParseError(token, f"Expected expression, found {token.kind} {token.lexeme!r}")
        self.advance()
        if token.kind == "INT":
            return node("Integer", token, value=token.lexeme)
        if token.kind in {"TRUE", "FALSE"}:
            return node("Boolean", token, value=token.kind == "TRUE")
        if token.kind in {"IDENT", "LENGTH"}:
            if self.take("LPAREN"):
                args = []
                if self.current.kind != "RPAREN":
                    args.append(self.expression())
                    while self.take("COMMA"):
                        args.append(self.expression())
                self.expect("RPAREN")
                return node("Call", token, name=token.lexeme, args=args)
            if token.kind == "LENGTH":
                raise ParseError(token, "length requires an argument list")
            return node("Name", token, name=token.lexeme)
        if token.kind == "LPAREN":
            value = self.expression()
            self.expect("RPAREN")
            return value
        if token.kind in {"FORALL", "EXISTS"}:
            self.expect("LPAREN")
            variable = self.declaration()
            self.expect("PIPE")
            predicate = self.expression()
            self.expect("RPAREN")
            return node("Quantifier", token, quantifier=token.lexeme, variable=variable, predicate=predicate)


def parse(text):
    try:
        tokens = tokenize(text)
    except LexError as error:
        return dict(ast=None, diagnostics=[error.diagnostic])
    parser = Parser(tokens)
    try:
        ast = parser.module()
    except RecursionError:
        parser.report(ParseError(parser.current, "Nesting is too deep for this parser"))
        ast = None
    return dict(ast=ast, diagnostics=parser.diagnostics)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Parse a program supplied as a string")
    source.add_argument("--file", type=Path, help="Parse a UTF-8 source file")
    args = cli.parse_args()
    try:
        text = args.text if args.text is not None else args.file.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        cli.exit(1, f"{error}\n")
    result = parse(text)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(1 if result["diagnostics"] else 0)


if __name__ == "__main__":
    main()
