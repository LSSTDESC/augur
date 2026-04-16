import numpy as np
import ast

SAFE_EVAL_GLOBALS = {'np': np}
SAFE_NUMPY_ATTRS = {name for name in dir(np) if not name.startswith('_')}


def _is_safe_ast(node):
    if isinstance(node, ast.Expression):
        return _is_safe_ast(node.body)
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_ast(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_safe_ast(k) and _is_safe_ast(v) for k, v in zip(node.keys, node.values))
    if isinstance(node, ast.BinOp):
        return _is_safe_ast(node.left) and _is_safe_ast(node.right) and \
               isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                    ast.Pow, ast.Mod, ast.FloorDiv))
    if isinstance(node, ast.UnaryOp):
        return _is_safe_ast(node.operand) and isinstance(node.op, (ast.UAdd, ast.USub))
    if isinstance(node, ast.BoolOp):
        return all(_is_safe_ast(v) for v in node.values) and isinstance(node.op, (ast.And, ast.Or))
    if isinstance(node, ast.Compare):
        return _is_safe_ast(node.left) and all(_is_safe_ast(comp) for comp in node.comparators) \
               and all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                       for op in node.ops)
    if isinstance(node, ast.Name):
        return node.id in SAFE_EVAL_GLOBALS
    if isinstance(node, ast.Attribute):
        return _is_safe_ast(node.value) and isinstance(node.attr, str) \
               and node.attr in SAFE_NUMPY_ATTRS
    if isinstance(node, ast.Call):
        return _is_safe_ast(node.func) and all(_is_safe_ast(arg) for arg in node.args) \
               and all(_is_safe_ast(kw.value) for kw in node.keywords)
    if isinstance(node, ast.Subscript):
        return _is_safe_ast(node.value) and _is_safe_ast(node.slice)
    if isinstance(node, ast.Slice):
        return (node.lower is None or _is_safe_ast(node.lower)) and \
               (node.upper is None or _is_safe_ast(node.upper)) and \
               (node.step is None or _is_safe_ast(node.step))
    return False


def _safe_eval(expr):
    if isinstance(expr, (list, dict, tuple, int, float, np.ndarray)):
        return expr
    tree = ast.parse(expr, mode='eval')
    if not _is_safe_ast(tree):
        raise ValueError(f"Unsafe expression: {expr}")
    return eval(compile(tree, '<safe_eval>', 'eval'), {'__builtins__': None}, SAFE_EVAL_GLOBALS)
