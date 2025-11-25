import sys
import numpy as np
import sympy as sp
from typing import List, Tuple, Callable
from data.Exampler_B import Example


def lambda_to_sympy(lambda_func: Callable, var_symbols: list, n: int) -> sp.Expr:
    test_values = var_symbols

    try:
        result = lambda_func(test_values)
        if isinstance(result, (sp.Basic, int, float)):
            return sp.sympify(result)
    except:
        pass

    print(f"Warning: Could not directly convert lambda to sympy, using reconstruction")

    return None


def sympy_to_lambda(expr: sp.Expr, var_symbols: list) -> Callable:
    f = sp.lambdify([var_symbols], expr, 'numpy')
    return lambda x: f(x)


class SymbolicExample:
    def __init__(self, n: int, D_zones, I_zones, U_zones, 
                 f_symbolic: List[sp.Expr],
                 var_symbols: list, 
                 name: str,
                 barrier_expr: sp.Expr = None,
                 zone_exprs: dict = None):
        
        self.n = n
        self.D_zones = np.array(D_zones)
        self.I_zones = np.array(I_zones)
        self.U_zones = np.array(U_zones)
        self.f_symbolic = f_symbolic
        self.var_symbols = var_symbols
        self.name = name

        self.barrier_expr = barrier_expr
        self.zone_exprs = zone_exprs

        self.f_lambda = [sympy_to_lambda(f_expr, var_symbols) for f_expr in f_symbolic]

    def to_regular_example(self) -> Example:
        return Example(
            n=self.n,
            D_zones=self.D_zones.tolist(),
            I_zones=self.I_zones.tolist(),
            U_zones=self.U_zones.tolist(),
            f=self.f_lambda,
            name=self.name
        )

    def print_symbolic(self):
        print(f"\nSymbolic System: {self.name}")
        print(f"Variables: {self.var_symbols}")
        for i, f_expr in enumerate(self.f_symbolic):
            print(f"  f[{i}] = {f_expr}")


def create_symbolic_example_from_expressions(n: int, D_zones, I_zones, U_zones,
                                             f_expressions: List[str], name: str) -> SymbolicExample:
    var_symbols = [sp.Symbol(f'x{i}') for i in range(n)]

    f_symbolic = []
    for expr_str in f_expressions:
        local_dict = {f'x{i}': var_symbols[i] for i in range(n)}
        expr = sp.sympify(expr_str, locals=local_dict)
        f_symbolic.append(expr)

    return SymbolicExample(n, D_zones, I_zones, U_zones, f_symbolic, var_symbols, name)


def apply_cdt_symbolic(example: SymbolicExample, A: np.ndarray, b: np.ndarray) -> SymbolicExample:
    n = example.n
    A_inv = np.linalg.inv(A)

    y_vars = [sp.Symbol(f'y{i}') for i in range(n)]

    x_exprs = []
    for i in range(n):
        expr = sum(A[i, j] * y_vars[j] for j in range(n)) + b[i]
        x_exprs.append(expr)

    F_of_Ay_plus_b = []
    for f_expr in example.f_symbolic:
        substitutions = {example.var_symbols[i]: x_exprs[i] for i in range(n)}
        f_substituted = f_expr.subs(substitutions)
        F_of_Ay_plus_b.append(f_substituted)

    f_prime_symbolic = []
    for i in range(n):
        expr = sum(A_inv[i, j] * F_of_Ay_plus_b[j] for j in range(n))
        expr = sp.simplify(expr)
        f_prime_symbolic.append(expr)

    def transform_zones_analytic(zones, A_inv: np.ndarray, b: np.ndarray, n: int) -> list:
        new_zones = []

        for i in range(n):
            a_inv_ii = A_inv[i, i]
            b_i = b[i]

            x_min, x_max = zones[i]

            y1 = a_inv_ii * (x_min - b_i)
            y2 = a_inv_ii * (x_max - b_i)

            new_y_min = min(y1, y2)
            new_y_max = max(y1, y2)

            new_zones.append([new_y_min, new_y_max])

        return new_zones

    new_D_zones = transform_zones_analytic(example.D_zones, A_inv, b, n)
    new_I_zones = transform_zones_analytic(example.I_zones, A_inv, b, n)
    new_U_zones = transform_zones_analytic(example.U_zones, A_inv, b, n)

    new_barrier_expr = None
    if example.barrier_expr is not None:
        substitutions = {example.var_symbols[i]: x_exprs[i] for i in range(n)}
        new_barrier_expr = example.barrier_expr.subs(substitutions)
        new_barrier_expr = sp.simplify(new_barrier_expr)

    new_example = SymbolicExample(
        n=n,
        D_zones=new_D_zones,
        I_zones=new_I_zones,
        U_zones=new_U_zones,
        f_symbolic=f_prime_symbolic,
        var_symbols=y_vars,
        name=f"{example.name}_cdt",
        barrier_expr=new_barrier_expr
    )

    return new_example


def apply_cfa_symbolic(example: SymbolicExample, B_expr: sp.Expr, S: np.ndarray, alpha: float) -> SymbolicExample:
    n = example.n

    grad_B = [sp.diff(B_expr, var) for var in example.var_symbols]

    S_grad_B = []
    for i in range(n):
        expr = sum(S[i, j] * grad_B[j] for j in range(n))
        S_grad_B.append(expr)

    f_prime_symbolic = []
    for i in range(n):
        expr = example.f_symbolic[i] + alpha * S_grad_B[i]
        expr = sp.simplify(expr)
        f_prime_symbolic.append(expr)

    new_barrier_expr = example.barrier_expr

    new_example = SymbolicExample(
        n=n,
        D_zones=example.D_zones,
        I_zones=example.I_zones,
        U_zones=example.U_zones,
        f_symbolic=f_prime_symbolic,
        var_symbols=example.var_symbols,
        name=f"{example.name}_cfa",
        barrier_expr=new_barrier_expr
    )

    return new_example


def create_simple_barrier(I_zones, U_zones, var_symbols):
    n = len(var_symbols)

    I_center = [(I_zones[i][0] + I_zones[i][1]) / 2 for i in range(n)]
    U_center = [(U_zones[i][0] + U_zones[i][1]) / 2 for i in range(n)]

    dist_I_sq = sum((var_symbols[i] - I_center[i])**2 for i in range(n))

    dist_U_sq = sum((var_symbols[i] - U_center[i])**2 for i in range(n))

    B_expr = dist_I_sq - dist_U_sq

    return sp.simplify(B_expr)


if __name__ == '__main__':
    pass