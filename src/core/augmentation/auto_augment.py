import os
import sys
import json
import time
import logging
import pickle
import numpy as np
import sympy as sp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
from tqdm import tqdm

from src.core.augmentation.augmentation_symbolic import (
    SymbolicExample,
    create_symbolic_example_from_expressions,
    apply_cdt_symbolic,
    apply_cfa_symbolic,
    create_simple_barrier
)
from data.Exampler_B import get_example_by_name, get_all_examples


class QualityController:

    @staticmethod
    def check_condition_number(A: np.ndarray, max_cond: float = 100.0) -> bool:
        try:
            cond = np.linalg.cond(A)
            return cond <= max_cond
        except:
            return False

    @staticmethod
    def check_zone_validity(system: SymbolicExample) -> bool:
        for i in range(system.n):
            I_min, I_max = system.I_zones[i]
            D_min, D_max = system.D_zones[i]
            if not (D_min <= I_min <= I_max <= D_max):
                return False

            U_min, U_max = system.U_zones[i]
            if not (D_min <= U_min <= U_max <= D_max):
                return False

            if not (U_max < I_min or I_max < U_min):
                return False

        return True

    @staticmethod
    def check_numerical_stability(system: SymbolicExample,
                                  num_points: int = 5) -> bool:
        for _ in range(num_points):
            x = np.array([
                np.random.uniform(system.D_zones[i][0], system.D_zones[i][1])
                for i in range(system.n)
            ])

            try:
                f_val = [f(x) for f in system.f_lambda]
                if not all(np.isfinite(f) for f in f_val):
                    return False
                if any(abs(f) > 1000 for f in f_val):
                    return False
            except:
                return False

        return True

    @staticmethod
    def check_expression_complexity(system: SymbolicExample, max_terms: int = 50) -> bool:
        for f_expr in system.f_symbolic:
            complexity = len(str(f_expr).split('+'))
            if complexity > max_terms:
                return False
        return True


class AugmentationEngine:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality = QualityController()

    def load_seed_examples(self) -> List[SymbolicExample]:
        seed_examples = []

        if 'use_builtin_examples' in self.config['input']:
            for name in self.config['input']['use_builtin_examples']:
                example = get_example_by_name(name)
                if example:
                    symbolic_ex = self._convert_to_symbolic(example)
                    seed_examples.append(symbolic_ex)

        if 'custom_examples' in self.config['input']:
            for ex_config in self.config['input']['custom_examples']:
                symbolic_ex = create_symbolic_example_from_expressions(
                    n=ex_config['n'],
                    D_zones=ex_config['D_zones'],
                    I_zones=ex_config['I_zones'],
                    U_zones=ex_config['U_zones'],
                    f_expressions=ex_config['f_expressions'],
                    name=ex_config['name']
                )
                seed_examples.append(symbolic_ex)

        return seed_examples

    def _convert_to_symbolic(self, example) -> Optional[SymbolicExample]:

        BUILTIN_EXPRESSION_MAP = {
            'C1': [
                "x1**2 - 5.5*x1",
                "-x0**2 + 6*x0"
            ],
            'C2': [
                "-x0 + x0*x1",
                "-x1 - x0*x1"
            ],
        }

        if example.name in BUILTIN_EXPRESSION_MAP:
            f_expressions = BUILTIN_EXPRESSION_MAP[example.name]

            symbolic_ex = create_symbolic_example_from_expressions(
                n=example.n,
                D_zones=example.D_zones.tolist(),
                I_zones=example.I_zones.tolist(),
                U_zones=example.U_zones.tolist(),
                f_expressions=f_expressions,
                name=example.name
            )

            if hasattr(example, 'barrier_expr_str') and example.barrier_expr_str:
                import sympy as sp
                local_dict = {f'x{i}': symbolic_ex.var_symbols[i] for i in range(example.n)}
                barrier_expr = sp.sympify(example.barrier_expr_str, locals=local_dict)
                symbolic_ex.barrier_expr = barrier_expr

            return symbolic_ex

        self.logger.warning(
            f"Example '{example.name}' does not have predefined symbolic expressions. "
            f"Please use 'custom_examples' in the config and provide 'f_expressions' directly."
        )
        return None
    
    def generate_cdt_variants(self, base_system: SymbolicExample,
                              num_variants: int) -> List[SymbolicExample]:
        variants = []
        cdt_config = self.config['cdt_transform']

        from tqdm import trange
        
        for i in trange(num_variants):
            A, b = self._generate_random_affine_transform(
                base_system.n,
                cdt_config['transform_params']
            )

            variant = apply_cdt_symbolic(base_system, A, b)
            variant.name = f"{base_system.name}_cdt_{i}"

            if self._passes_quality_checks(variant):
                variants.append(variant)

        return variants

    def generate_cfa_variants(self, base_system: SymbolicExample,
                              num_variants: int) -> List[SymbolicExample]:
        variants = []
        cfa_config = self.config['cfa_transform']

        if base_system.barrier_expr is not None:
            B_expr = base_system.barrier_expr
        else:
            B_expr = create_simple_barrier(
                base_system.I_zones,
                base_system.U_zones,
                base_system.var_symbols
            )

        for i in range(num_variants):
            S = self._generate_random_skew_symmetric(
                base_system.n,
                cfa_config.get('skew_symmetric_params')
            )
            alpha = np.random.uniform(
                cfa_config['alpha_range'][0],
                cfa_config['alpha_range'][1]
            )

            variant = apply_cfa_symbolic(base_system, B_expr, S, alpha)
            variant.name = f"{base_system.name}_cfa_{i}"

            if self._passes_quality_checks(variant):
                variants.append(variant)

        return variants

    def _generate_random_affine_transform(self, n: int, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        diag_elements = np.random.uniform(
            params['A_scale_range'][0],
            params['A_scale_range'][1],
            n
        )

        diag_elements = np.where(np.abs(diag_elements) < 0.1, 0.5, diag_elements)

        A = np.diag(diag_elements)

        b = np.random.uniform(params['b_range'][0], params['b_range'][1], n)

        return A, b

    def _generate_random_skew_symmetric(self, n: int, params: Optional[Dict] = None) -> np.ndarray:
        if params is None:
            params = {'scale_range': [0.1, 0.5]}

        M = np.random.randn(n, n) * params.get('scale_range', [0.1, 0.5])[1]
        S = (M - M.T) / 2

        scale = np.random.uniform(params.get('scale_range', [0.1, 0.5])[0],
                                  params.get('scale_range', [0.1, 0.5])[1])
        S = S * scale

        return S

    def _passes_quality_checks(self, system: SymbolicExample) -> bool:
        qc_config = self.config.get('quality_control', {})

        return True


class DataAugmentator:

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.engine = AugmentationEngine(self.config)
        self._setup_logging()
        self._setup_directories()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_logging(self):
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))

        log_file = log_config.get('file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file) if log_file else logging.NullHandler(),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger('DataAugmentator')

    def _setup_directories(self):
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ['logs', 'cache', 'results']:
            dir_path = output_dir.parent / subdir
            dir_path.mkdir(exist_ok=True)

    def generate_dataset(self) -> List[SymbolicExample]:
        self.logger.info("="*80)
        self.logger.info("Starting automated data augmentation process")
        self.logger.info("="*80)

        self.logger.info("Loading seed examples...")
        seed_examples = self.engine.load_seed_examples()
        self.logger.info(f"Successfully loaded {len(seed_examples)} seed examples")

        all_systems = list(seed_examples)

        if self.config['cdt_transform']['enabled']:
            self.logger.info("Generating CDT variants...")
            cdt_variants = self._generate_all_cdt_variants(seed_examples)
            all_systems.extend(cdt_variants)
            self.logger.info(f"Generated {len(cdt_variants)} CDT variants")

        if self.config['cfa_transform']['enabled']:
            self.logger.info("Generating CFA variants...")
            base_systems = [seed_examples[0]] if seed_examples else []
            cfa_variants = self._generate_all_cfa_variants(base_systems)
            all_systems.extend(cfa_variants)
            self.logger.info(f"Generated {len(cfa_variants)} CFA variants")

        self.logger.info("Saving results...")
        self._save_results(all_systems)

        self.logger.info("="*80)
        self.logger.info(f"Data augmentation completed! Generated a total of {len(all_systems)} systems")
        self.logger.info("="*80)

        return all_systems

    def _generate_all_cdt_variants(self, seed_examples: List[SymbolicExample]) -> List[SymbolicExample]:
        all_variants = []
        cdt_config = self.config['cdt_transform']

        if 'random_seed' in cdt_config:
            np.random.seed(cdt_config['random_seed'])

        for seed_ex in seed_examples:
            print(seed_ex.name)
            variants = self.engine.generate_cdt_variants(
                seed_ex,
                cdt_config['num_variants_per_example']
            )
            all_variants.extend(variants)
        
        return all_variants

    def _generate_all_cfa_variants(self, base_systems: List[SymbolicExample]) -> List[SymbolicExample]:
        all_variants = []
        cfa_config = self.config['cfa_transform']

        for base_ex in base_systems:
            variants = self.engine.generate_cfa_variants(
                base_ex,
                cfa_config['num_variants_per_base']
            )
            all_variants.extend(variants)

        return all_variants

    def _convert_expr_to_x_vars(self, expr: sp.Expr) -> str:
        symbols = expr.free_symbols

        substitutions = {}
        for sym in symbols:
            sym_name = str(sym)
            if sym_name.startswith('y') and sym_name[1:].isdigit():
                idx = sym_name[1:]
                x_var = sp.Symbol(f'x{idx}')
                substitutions[sym] = x_var

        if substitutions:
            expr = expr.subs(substitutions)

        return str(expr)

    def _save_results(self, systems: List[SymbolicExample]):
        output_config = self.config['output']
        output_dir = Path(output_config['output_dir'])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = output_config.get('file_naming', {}).get('prefix', 'dataset')

        if output_config.get('save_statistics', False):
            stats = self._compute_statistics(systems)
            stats_file = output_dir / f"{prefix}_statistics_{timestamp}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

        formats = output_config['formats']

        if 'txt' in formats:
            txt_file = output_dir / f"{prefix}_{timestamp}.txt"
            self._save_as_txt(systems, txt_file)

        if 'json' in formats:
            json_file = output_dir / f"{prefix}_{timestamp}.json"
            self._save_as_json(systems, json_file)

        if 'pickle' in formats:
            serializable_systems = []
            for sys in systems:
                serializable_sys = {
                    'name': sys.name,
                    'n': sys.n,
                    'D_zones': sys.D_zones.tolist(),
                    'I_zones': sys.I_zones.tolist(),
                    'U_zones': sys.U_zones.tolist(),
                    'f_expressions': [self._convert_expr_to_x_vars(expr) for expr in sys.f_symbolic],
                    'var_symbols': [str(sym) for sym in sys.var_symbols]
                }

                if sys.barrier_expr is not None:
                    serializable_sys['barrier_expr'] = self._convert_expr_to_x_vars(sys.barrier_expr)

                serializable_systems.append(serializable_sys)

            pickle_file = output_dir / f"{prefix}_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(serializable_systems, f)

    def _compute_statistics(self, systems: List[SymbolicExample]) -> Dict[str, Any]:
        stats = {
            'total_systems': len(systems),
            'cdt_systems': len([s for s in systems if 'cdt' in s.name]),
            'cfa_systems': len([s for s in systems if 'cfa' in s.name]),
            'original_systems': len([s for s in systems if 'cdt' not in s.name and 'cfa' not in s.name]),
            'dimensions': list(set(s.n for s in systems)),
            'generation_time': datetime.now().isoformat()
        }
        return stats

    def _save_as_txt(self, systems: List[SymbolicExample], file_path: Path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for system in systems:
                f.write(f"\n{'='*60}\n")
                f.write(f"System: {system.name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Dimension: {system.n}\n")
                f.write(f"Domain: {system.D_zones.tolist()}\n")
                f.write(f"Initial: {system.I_zones.tolist()}\n")
                f.write(f"Unsafe: {system.U_zones.tolist()}\n")
                f.write(f"\nDynamics:\n")
                for i, f_expr in enumerate(system.f_symbolic):
                    f.write(f"  f[{i}] = {self._convert_expr_to_x_vars(f_expr)}\n")

                if system.barrier_expr is not None:
                    f.write(f"\nBarrier Function:\n")
                    f.write(f"  B = {self._convert_expr_to_x_vars(system.barrier_expr)}\n")

                f.write("\n")

    def _save_as_json(self, systems: List[SymbolicExample], file_path: Path):
        data = []
        for system in systems:
            system_data = {
                'name': system.name,
                'n': system.n,
                'D_zones': system.D_zones.tolist(),
                'I_zones': system.I_zones.tolist(),
                'U_zones': system.U_zones.tolist(),
                'f_expressions': [self._convert_expr_to_x_vars(expr) for expr in system.f_symbolic]
            }

            if system.barrier_expr is not None:
                system_data['barrier_expr'] = self._convert_expr_to_x_vars(system.barrier_expr)

            data.append(system_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Automated Data Augmentation Tool')
    parser.add_argument('--config', '-c',
                        default='config/auto_augment_example.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', '-o',
                        help='Output directory (optional, overrides config)')

    args = parser.parse_args()

    augmentator = DataAugmentator(args.config)

    systems = augmentator.generate_dataset()

    print("\n" + "="*80)
    print("Data Augmentation Summary")
    print("="*80)
    print(f"Total systems: {len(systems)}")
    print(f"Output directory: {augmentator.config['output']['output_dir']}")
    print("="*80)


if __name__ == '__main__':
    main()