#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('.')
sys.path.append('./')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping visualization features")

PRESET_CONFIGS = {
    'small': {
        'description': 'Small dataset (~50 systems)',
        'config': {
            'input': {'use_builtin_examples': ['C1', 'C2']},
            'cdt_transform': {
                'enabled': True,
                'num_variants_per_example': 5,
                'transform_params': {
                    'A_scale_range': [0.5, 2.0],
                    'A_diagonal_multiplier': 2.0,
                    'b_range': [-1.0, 1.0]
                }
            },
            'cfa_transform': {
                'enabled': True,
                'num_variants_per_base': 3,
                'alpha_range': [0.1, 0.5]
            },
            'output': {
                'output_dir': 'results/augmented_systems_small',
                'formats': ['txt', 'json']
            },
            'quality_control': {'validate_zones': True},
            'parallelization': {'enabled': False}
        }
    },
    'medium': {
        'description': 'Medium dataset (~500 systems)',
        'config': {
            'input': {'use_builtin_examples': ['C1', 'C2', 'C3']},
            'cdt_transform': {
                'enabled': True,
                'num_variants_per_example': 15,
                'transform_params': {
                    'A_scale_range': [0.5, 2.0],
                    'A_diagonal_multiplier': 2.0,
                    'b_range': [-1.0, 1.0]
                }
            },
            'cfa_transform': {
                'enabled': True,
                'num_variants_per_base': 8,
                'alpha_range': [0.05, 0.5]
            },
            'output': {
                'output_dir': 'results/augmented_systems_medium',
                'formats': ['txt', 'json']
            },
            'quality_control': {'validate_zones': True},
            'parallelization': {'enabled': True, 'max_workers': 4}
        }
    },
    'large': {
        'description': 'Large dataset (~5000+ systems)',
        'config': {
            'input': {
                'use_builtin_examples': ['C1', 'C2', 'C3'],
                'custom_examples': [
                    {
                        'name': 'Custom1',
                        'n': 2,
                        'D_zones': [[-5, 5], [-5, 5]],
                        'I_zones': [[1, 2], [1, 2]],
                        'U_zones': [[-3, -2], [-3, -2]],
                        'f_expressions': ['x0 + x1', 'x0**2 - x1']
                    },
                    {
                        'name': 'Custom2',
                        'n': 3,
                        'D_zones': [[-3, 3]] * 3,
                        'I_zones': [[1, 1.5]] * 3,
                        'U_zones': [[-2, -1]] * 3,
                        'f_expressions': ['x0*x1 - x2', 'x1**2 - x0', 'sin(x0) - x2']
                    }
                ]
            },
            'cdt_transform': {
                'enabled': True,
                'num_variants_per_example': 25,
                'transform_params': {
                    'A_scale_range': [0.5, 2.0],
                    'A_diagonal_multiplier': 2.0,
                    'b_range': [-1.0, 1.0]
                }
            },
            'cfa_transform': {
                'enabled': True,
                'num_variants_per_base': 12,
                'alpha_range': [0.05, 0.5]
            },
            'output': {
                'output_dir': 'results/augmented_systems_large',
                'formats': ['json', 'pickle']
            },
            'quality_control': {
                'validate_zones': True,
                'numerical_stability': {
                    'enabled': True,
                    'test_points_per_system': 5,
                    'max_value': 1000.0
                },
                'deduplication': True
            },
            'parallelization': {'enabled': True, 'max_workers': 8},
            'progress_tracking': {'enabled': True, 'progress_bar': True}
        }
    }
}


def create_config_file(size: str, cdt_count: int = None, cfa_count: int = None,
                       output_dir: str = None) -> Path:

    if size in PRESET_CONFIGS:
        config = PRESET_CONFIGS[size]['config'].copy()
    else:
        config = {
            'input': {'use_builtin_examples': ['C1', 'C2']},
            'cdt_transform': {
                'enabled': True,
                'num_variants_per_example': 5,
                'transform_params': {
                    'A_scale_range': [0.5, 2.0],
                    'A_diagonal_multiplier': 2.0,
                    'b_range': [-1.0, 1.0]
                }
            },
            'cfa_transform': {
                'enabled': True,
                'num_variants_per_base': 3,
                'alpha_range': [0.1, 0.5]
            },
            'output': {
                'output_dir': 'results/augmented_systems_custom',
                'formats': ['txt', 'json']
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/quick_augment.log',
                'console_output': True
            }
        }

    if cdt_count is not None:
        config['cdt_transform']['num_variants_per_example'] = cdt_count

    if cfa_count is not None:
        config['cfa_transform']['num_variants_per_base'] = cfa_count

    if output_dir:
        config['output']['output_dir'] = output_dir

    config['logging'] = {
        'level': 'INFO',
        'file': 'logs/quick_augment.log',
        'console_output': True
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = PROJECT_ROOT / f"config/quick_augment_{timestamp}.yaml"

    config_file.parent.mkdir(exist_ok=True)

    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)

    return config_file


def visualize_2d_systems(systems, output_dir: Path, max_plots: int = 6):
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not installed, skipping visualization")
        return

    systems_2d = [s for s in systems if s.n == 2]

    if not systems_2d:
        print("ℹ️  No 2D systems found, skipping visualization")
        return

    print(f"\n📊 Generating vector field visualizations ({min(len(systems_2d), max_plots)} systems)...")

    plot_systems = systems_2d[:max_plots]

    n_plots = len(plot_systems)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, (ax, system) in enumerate(zip(axes[:n_plots], plot_systems)):
        x1_min, x1_max = system.D_zones[0]
        x2_min, x2_max = system.D_zones[1]

        x1 = np.linspace(x1_min, x1_max, 20)
        x2 = np.linspace(x2_min, x2_max, 20)
        X1, X2 = np.meshgrid(x1, x2)

        U = np.zeros_like(X1)
        V = np.zeros_like(X2)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                point = np.array([X1[i, j], X2[i, j]])
                try:
                    f_val = [f(point) for f in system.f_lambda]
                    U[i, j] = f_val[0]
                    V[i, j] = f_val[1]
                except:
                    U[i, j] = 0
                    V[i, j] = 0

        ax.quiver(X1, X2, U, V, alpha=0.6, scale=None, scale_units='xy')

        I_x1_min, I_x1_max = system.I_zones[0]
        I_x2_min, I_x2_max = system.I_zones[1]
        ax.add_patch(plt.Rectangle((I_x1_min, I_x2_min),
                                   I_x1_max - I_x1_min,
                                   I_x2_max - I_x2_min,
                                   fill=True, color='green', alpha=0.3,
                                   edgecolor='darkgreen', linewidth=2, label='Initial'))

        U_x1_min, U_x1_max = system.U_zones[0]
        U_x2_min, U_x2_max = system.U_zones[1]
        ax.add_patch(plt.Rectangle((U_x1_min, U_x2_min),
                                   U_x1_max - U_x1_min,
                                   U_x2_max - U_x2_min,
                                   fill=True, color='red', alpha=0.3,
                                   edgecolor='darkred', linewidth=2, label='Unsafe'))

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel('x₀', fontsize=12)
        ax.set_ylabel('x₁', fontsize=12)
        ax.set_title(f'{system.name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal', adjustable='box')

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    vis_file = output_dir / 'vector_field_visualization.png'
    plt.savefig(vis_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualization saved: {vis_file}")


def run_augmentation(config_file: Path, enable_visualization: bool = True):
    from src.core.augmentation.auto_augment import DataAugmentator

    print("\n" + "="*80)
    print("Starting Automated Data Augmentation")
    print("="*80)
    print(f"Config File: {config_file}")
    print(f"Output Directory: {config_file.parent.parent}/results/")
    print("="*80 + "\n")

    try:
        augmentator = DataAugmentator(str(config_file))
        systems = augmentator.generate_dataset()

        print("\n" + "="*80)
        print("Data Augmentation Summary")
        print("="*80)
        print(f"✅ Total Systems: {len(systems)}")
        print(f"📁 Output Directory: {augmentator.config['output']['output_dir']}")
        print(f"📊 Output Formats: {', '.join(augmentator.config['output']['formats'])}")
        print("="*80 + "\n")

        if enable_visualization and systems:
            output_dir = Path(augmentator.config['output']['output_dir'])
            visualize_2d_systems(systems, output_dir, max_plots=6)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Quick Automated Data Augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  %(prog)s --size small      # Generate small dataset
  %(prog)s --size medium     # Generate medium dataset
  %(prog)s --size large      # Generate large dataset
  %(prog)s --custom --cdt 10 --cfa 5  # Custom parameters

Preset Configuration Details:
  small:  ~50 systems, uses C1, C2, 5 CDT variants/system, 3 CFA variants/system
  medium: ~500 systems, uses C1, C2, C3, 15 CDT variants/system, 8 CFA variants/system
  large:  ~5000+ systems, uses C1, C2, C3 + 2 custom systems, 25 CDT variants/system, 12 CFA variants/system
        """
    )

    parser.add_argument('--size', choices=['small', 'medium', 'large'],
                        help='Dataset size preset')

    parser.add_argument('--custom', action='store_true',
                        help='Use custom parameters')
    parser.add_argument('--cdt', type=int, default=10,
                        help='Number of CDT variants per seed example (default: 10)')
    parser.add_argument('--cfa', type=int, default=5,
                        help='Number of CFA variants per base system (default: 5)')

    parser.add_argument('--output', '-o', type=str,
                        help='Output directory (optional)')

    parser.add_argument('--config-only', action='store_true',
                        help='Generate config file only, do not run augmentation')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization (even for 2D systems)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Creating Configuration")
    print("="*80)

    if args.size:
        print(f"Using preset: {args.size}")
        print(f"Description: {PRESET_CONFIGS[args.size]['description']}")
        config_file = create_config_file(args.size)
    elif args.custom:
        print(f"Custom configuration: CDT={args.cdt}, CFA={args.cfa}")
        config_file = create_config_file(
            'custom',
            cdt_count=args.cdt,
            cfa_count=args.cfa,
            output_dir=args.output
        )
    else:
        print("Error: Please specify --size or --custom")
        parser.print_help()
        return

    print(f"Config file created: {config_file}")

    if args.config_only:
        print("\n✅ Configuration complete! Run augmentation using:")
        print(f"python {sys.argv[0]} --config {config_file}")
        return

    success = run_augmentation(config_file, enable_visualization=not args.no_viz)

    if success:
        print("\n" + "="*80)
        print("🎉 Successfully Completed!")
        print("="*80)
        print("\nView Results:")
        print(f"  - Text Files: {Path(config_file).parent.parent}/results/")
        print(f"  - Log Files: {config_file.parent.parent}/logs/")
    else:
        print("\n❌ Augmentation Failed. Check log files.")
        sys.exit(1)


if __name__ == '__main__':
    main()