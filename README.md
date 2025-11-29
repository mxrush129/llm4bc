
## Overview

LLM4BC is a data augmentation tool specifically designed to enhance the capabilities of large language models (LLMs) in solving barrier certificates for dynamical systems. It enables the batch generation of large-scale, diverse augmented dynamical systems using only a small number of samples. During the data augmentation process, Coordinate Diffeomorphism Transformation (CDT) technology ensures the efficient production of rich variants, thereby significantly improving the model's generalization performance and solving accuracy.

## Installation

### Prerequisites

- Operating System: Linux Ubuntu 22.04.5 LTS (GNU/Linux 6.8.0-87-generic x86_64)
- Python: >= 3.11

### Install from Source


```bash
git clone https://github.com/mxrush129/llm4bc.git
cd llm4bc
pip install -r requirements.txt
```

## Usage

### Quickstart
Run the following command to generate a sample augmented dataset:

```bash
python quick_augment.py --size small 
```
The results will be output to the `results/augmented_systems_small` directory, which contains two main files:

* **`dataset_<timestamp>.txt`**: Detailed execution logs. (Note: If the file size is excessive, it will be automatically converted to `.pkl` format).
* **`dataset_<timestamp>.json`**: The generated augmented dataset.

Below is a sample entry from the augmented dataset:

```txt
{
  "name": "C1_cdt_1",
  "n": 2,  # System dimension
  "D_zones": [[0.40749882677907323, 2.5185794201130696],
      [1.6225885178345565, 5.320109187814466]
  ], # System domain, format: [[lb1, lb2], [ub1, ub2]]
  "I_zones": [
      [1.9908092717795705, 2.2546943459463202],
      [1.530150501085059, 1.7150265345840543]
  ], # Initial set
  "U_zones": [
      [0.40749882677907323, 0.9352689751125723],
      [2.5469686853295337, 3.471348852824511]
  ], # Unsafe set
  "f_expressions": [
      "0.617651618470489*x1**2 - 4.00269602117782*x1 + 2.49361377157736",
      "-3.31864635401627*x0**2 + 9.71061890561493*x0 + 1.21591382155993"
  ], # System differential equations (Vector Field)
  "barrier_expr": "-13.2338066019222*x0**2 - ... + 12.8832685534349" # Barrier Certificate
}
```

### Advanced Usage

Through command-line arguments, you can flexibly control dataset size, augmentation intensity, and output behavior.

#### 1. Arguments

| Argument | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--size` | - | `str` | - | Selects a dataset size preset (`small`, `medium`, `large`). Mutually exclusive with `--custom`. |
| `--custom` | - | `flag` | `False` | Enables custom parameter mode. Use with `--cdt` and `--cfa`. |
| `--cdt` | - | `int` | `10` | **CDT Variants**: Number of Coordinate Transformation variants per seed example. |
| `--cfa` | - | `int` | `5` | **CFA Variants**: Number of Field Augmentation variants per base system. |
| `--output` | `-o` | `str` | - | Specifies the output directory (optional). |
| `--no-viz` | - | `flag` | `False` | Disables visualization plotting. Recommended for server environments or large-scale generation. |
| `--config-only`| - | `flag` | `False` | Generates the configuration file only, without running the augmentation. Useful for debugging. |

#### 2. Usage Examples

**Scenario A: Using Presets (Recommended)**
Quickly generate datasets of standard scales:
* **Small (~50 systems)**: `python quick_augment.py --size small`
* **Medium (~500 systems)**: `python quick_augment.py --size medium`
* **Large (~5000+ systems)**: `python quick_augment.py --size large --no-viz`

**Scenario B: Custom Augmentation Intensity**
To generate data with specific variant density (e.g., 20 coordinate transformations per seed, 8 field distortions per transformation):

```bash
python quick_augment.py --custom --cdt 20 --cfa 8 --output ./results/my_custom_set
```
#### 3. Advanced Customization via YAML

For precise control over transformation parameters (e.g., matrix scaling ranges, distortion intensity) or to ensure reproducibility, it is recommended to use a YAML configuration file.

**Step 1: Create a Configuration File**
Create a new YAML file (e.g., `custom_config.yaml`) in the `./config` directory with the following structure:

```yaml
input:
  use_builtin_examples: [C1, C2, C3] # Specify base examples to use

cdt_transform:
  enabled: true
  num_variants_per_example: 15       # Number of CDT variants per seed example
  transform_params:
    A_scale_range: [0.5, 2.0]        # Scaling range for transformation matrix A
    A_diagonal_multiplier: 2.0       # Diagonal reinforcement multiplier
    b_range: [-1.0, 1.0]             # Range for bias vector b

cfa_transform:
  enabled: true
  num_variants_per_base: 8           # Number of CFA variants per CDT variant
  alpha_range: [0.05, 0.5]           # Intensity of vector field distortion

output:
  output_dir: "results/augmented_systems_medium"
  formats: ["txt", "json"]
  save_statistics: true

logging:
  level: "INFO"
  file: "logs/quick_augment.log"
  console_output: true

quality_control:
  validate_zones: true               # Enable domain validity checks

parallelization:
  enabled: true
  max_workers: 4                     # Number of parallel worker processes

progress_tracking:
  enabled: true
  progress_bar: true
```
