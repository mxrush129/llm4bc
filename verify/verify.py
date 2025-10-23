import json
import sympy as sp
import time
from SumOfSquares import SOSProblem # 只导入 SOSProblem，因为我们将手动构建所有多项式
from functools import reduce
from itertools import product

class BarrierValidator:
    """
    A class to validate a given Barrier Certificate (B).
    This version correctly implements the S-procedure by manually creating 
    polynomial multipliers with decision variables, aligning with the user's original logic.
    """
    def __init__(self, data_str: str):
        # print("--- initializing Validator ---")
        self.var_count = 0  # Counter for decision variables
        
        # 1. Parse all data from the JSON string into sympy expressions
        parsed_data = self._parse_input(data_str)
        self.x_sympy = parsed_data['vars']
        self.n = len(self.x_sympy)
        self.f_sympy = parsed_data['f']
        self.B_sympy = parsed_data['B']
        self.g_I_sympy = parsed_data['g_I_sphere']
        self.g_U_sympy = parsed_data['g_U_sphere']
        self.g_D_rect_sympy = parsed_data['g_D_rect']
        
        # NOTE: Unlike previous attempts, we do not convert to a new variable type here.
        # We will work with sympy expressions and only convert the final SOS expression.
        
        # Calculate Lie Derivative in sympy
        grad_B = [sp.diff(self.B_sympy, var) for var in self.x_sympy]
        self.L_f_B_sympy = sp.expand(sum(grad_B[i] * self.f_sympy[i] for i in range(self.n)))

        # print(f"Barrier Certificate B(x): {self.B_sympy}")
        # print(f"Lie Derivative L_f B(x): {sp.simplify(self.L_f_B_sympy)}\n")

    def _parse_input(self, data_str: str):
        # This function remains the same, it correctly parses into sympy objects
        data = json.loads(json.loads(data_str)['input'])
        generated_output = json.loads(data_str)['generated_output']
        var_names = data['variables']
        vars_sympy = sp.symbols(var_names)
        f_sympy = [sp.sympify(s) for s in data['system_dynamics_str']]
        B_sympy = sp.sympify(generated_output)
        init_set = data['initial_set']
        center_I = [sp.sympify(c) for c in init_set['center']]
        radius_I = sp.sympify(init_set['radius'])
        dist_sq_I = sum((vars_sympy[i] - center_I[i])**2 for i in range(len(vars_sympy)))
        g_I_sphere = radius_I**2 - dist_sq_I
        unsafe_set = data['unsafe_set']
        center_U = [sp.sympify(c) for c in unsafe_set['center']]
        radius_U = sp.sympify(unsafe_set['radius'])
        dist_sq_U = sum((vars_sympy[i] - center_U[i])**2 for i in range(len(vars_sympy)))
        g_U_sphere = radius_U**2 - dist_sq_U
        domain = data['domain']
        bounds = [[sp.sympify(v) for v in b] for b in domain['bounds']]
        g_D_rect = [(vars_sympy[i] - b[0]) * (b[1] - vars_sympy[i]) for i, b in enumerate(bounds)]
        
        return {
            "vars": vars_sympy, "f": f_sympy, "B": B_sympy,
            "g_I_sphere": g_I_sphere, "g_U_sphere": g_U_sphere, "g_D_rect": g_D_rect,
        }

    def _create_polynomial(self, prob, deg=2, is_sos=False):
        """
        Creates a general polynomial with decision variables as coefficients.
        If is_sos=True, also adds a constraint that this polynomial must be SOS.
        This is the correct, manual way to create multipliers.
        """
        poly = 0
        monomials = []
        exponents = [e for e in product(range(deg + 1), repeat=self.n) if sum(e) <= deg]
        for e in exponents:
            # Create the term (e.g., x1**2 * x2)
            term = reduce(lambda a, b: a * b, [self.x_sympy[i] ** exp for i, exp in enumerate(e)], 1)
            monomials.append(term)
            
            # Create a decision variable for its coefficient
            # Using sympy.Symbol for the coefficient
            coeff = sp.symbols('c' + str(self.var_count))
            self.var_count += 1
            poly += coeff * term
            
        if is_sos:
            # The SumOfSquares library works with string expressions and variable lists
            prob.add_sos_constraint(str(poly), self.x_sympy)
            
        return poly

    def verify_initial_set(self, multiplier_deg=2, epsilon=1e-6):
        prob = SOSProblem()
        # Create an SOS multiplier s(x)
        s_I = self._create_polynomial(prob, multiplier_deg, is_sos=True)
        # Build the final expression in sympy
        sos_expr_sympy = -(self.B_sympy + epsilon) - s_I * self.g_I_sympy
        sos_expr_sympy = sp.expand(sos_expr_sympy)
        # Add the main constraint to the problem
        prob.add_sos_constraint(sos_expr_sympy, self.x_sympy)
        try:
            prob.solve(solver='mosek')
            return True
        except Exception:
            return False

    def verify_unsafe_set(self, multiplier_deg=2, epsilon=1e-6):
        prob = SOSProblem()
        s_U = self._create_polynomial(prob, multiplier_deg, is_sos=True)
        sos_expr_sympy = (self.B_sympy - epsilon) - s_U * self.g_U_sympy
        sos_expr_sympy = sp.expand(sos_expr_sympy)
        prob.add_sos_constraint(str(sos_expr_sympy), self.x_sympy)
        try:
            prob.solve(solver='mosek')
            return True
        except Exception:
            return False

    def verify_lie_derivative(self, sos_multiplier_deg=2, lambda_deg=2):
        prob = SOSProblem()
        # Create SOS multipliers s_i(x) for the domain
        s_D_multipliers = [self._create_polynomial(prob, sos_multiplier_deg, is_sos=True) for _ in range(self.n)]
        # Create a general polynomial multiplier lambda(x) (not necessarily SOS)
        lambda_poly = self._create_polynomial(prob, lambda_deg, is_sos=False)
        
        sos_expr_sympy = -self.L_f_B_sympy - lambda_poly * self.B_sympy
        for i in range(self.n):
            sos_expr_sympy -= s_D_multipliers[i] * self.g_D_rect_sympy[i]
        
        sos_expr_sympy = sp.expand(sos_expr_sympy)
        prob.add_sos_constraint(str(sos_expr_sympy), self.x_sympy)
        
        try:
            prob.solve(solver='mosek')
            return True
        except Exception as e:
            print(f"   Solver Error: {e}")
            return False

    import time

    def verify_all(self, degs={'init': 2, 'unsafe': 2, 'lie_s': 2, 'lie_lambda': 2}, verbose=True):
        """
        Verifies all three conditions for a valid Barrier Certificate.
        
        Args:
            degs (dict): A dictionary specifying the degrees for the SOS multipliers.
            verbose (bool): If True, prints detailed step-by-step results of the verification.
        
        Returns:
            bool: True if all conditions are met, False otherwise.
        """
        t_start = time.time()
        
        if verbose:
            print("--- Running All Verification Checks ---")

        # 1. Verify Initial Set Condition
        if verbose:
            print("\n1. Verifying Initial Set Condition (B(x) <= 0 for x in X_init)...")
        init_ok = self.verify_initial_set(multiplier_deg=degs['init'])
        if verbose:
            print(f"   Result: {'✅ VERIFIED' if init_ok else '❌ FAILED'}")
        
        if not init_ok:
            if verbose:
                print("\nValidation stopped: Initial Set condition failed.")
            return False

        # 2. Verify Unsafe Set Condition
        if verbose:
            print("\n2. Verifying Unsafe Set Condition (B(x) > 0 for x in X_unsafe)...")
        unsafe_ok = self.verify_unsafe_set(multiplier_deg=degs['unsafe'])
        if verbose:
            print(f"   Result: {'✅ VERIFIED' if unsafe_ok else '❌ FAILED'}")

        if not unsafe_ok:
            if verbose:
                print("\nValidation stopped: Unsafe Set condition failed.")
            return False
        
        # 3. Verify Lie Derivative Condition
        if verbose:
            print("\n3. Verifying Lie Derivative Condition (L_f B(x) <= 0 for B(x) = 0)...")
        lie_ok = self.verify_lie_derivative(sos_multiplier_deg=degs['lie_s'], lambda_deg=degs['lie_lambda'])
        if verbose:
            print(f"   Result: {'✅ VERIFIED' if lie_ok else '❌ FAILED'}")

        if not lie_ok:
            if verbose:
                print("\nValidation stopped: Lie Derivative condition failed.")
            return False
        
        t_end = time.time()
        
        if verbose:
            print("\n" + "="*30)
            print("🎉 All conditions VERIFIED.")
            print(f"Total validation time: {round(t_end - t_start, 2)} s")
            print("="*30)
            
        return True

if __name__ == '__main__':
    dataset_string = """
    {"instruction": "", "input": "{\\"system_dynamics_str\\":[\\"-2.55241476715261*x1 - 15.2379261642369*x2 + 5.44758523284739\\",\\"12.3427556985422*x1 - 13.4475852328474*x2 - 1.44758523284739\\"],\\"domain\\":{\\"type\\":\\"rectangle\\",\\"bounds\\":[[\\"-5.000\\",\\"5.000\\"],[\\"-5.000\\",\\"5.000\\"]]},\\"initial_set\\":{\\"type\\":\\"sphere\\",\\"center\\":[\\"0.4286\\",\\"0.2857\\"],\\"radius\\":\\"0.5000\\"},\\"unsafe_set\\":{\\"type\\":\\"sphere\\",\\"center\\":[\\"5.000\\",\\"-5.000\\"],\\"radius\\":\\"0.5000\\"},\\"variables\\":[\\"x1\\",\\"x2\\"]}", "generated_output": "3*x1**2 - 2*x1*x2 - 2*x1 + 5*x2**2 - 2*x2"}
    """
    #2.6166*x1**2 - 3.2906*x1*x2 - 0.0124*x1 + 0.7579*x2**2 + 0.9985*x2 - 0.3316
    try:
        validator = BarrierValidator(dataset_string)
        is_valid = validator.verify_all(degs={'init': 2, 'unsafe': 2, 'lie_s': 2, 'lie_lambda': 2}, verbose=False)
        print("\n-------------------------------------------")
        if is_valid:
            print("✅ Conclusion: The provided polynomial IS a valid Barrier Certificate.")
        else:
            print("❌ Conclusion: The provided polynomial IS NOT a valid Barrier Certificate.")
        print("-------------------------------------------")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")