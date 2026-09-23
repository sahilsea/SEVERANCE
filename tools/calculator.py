"""Safe arithmetic expression evaluator -- a deterministic, hand-verified
tool the agent CALLS, never something an LLM computes itself and is simply
trusted on. Small local models get arithmetic wrong often enough that this
matters: the model's only job is to extract the expression, not solve it.

No eval()/exec(): expressions are parsed via Python's ast module and only
numeric literals, +-*/ // % **, unary +/-, and parentheses are permitted --
no names, no calls, no attribute access, so there is no code-injection
surface here at all (unlike tools/sandbox.py, which deliberately runs real
code in an isolated subprocess for that reason).
"""

from __future__ import annotations

import ast
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(Exception):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError(f"Unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalculatorError("Division by zero.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Expression contains something other than numbers and arithmetic operators ({type(node).__name__}).")


def calculate(expression: str) -> float:
    """Evaluate a pure arithmetic expression. Raises CalculatorError (never
    a raw Python exception) on anything unparseable or unsafe, so callers
    can surface a clean message instead of a stack trace."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Could not parse '{expression}' as an arithmetic expression: {e.msg}")
    return _eval_node(tree.body)
