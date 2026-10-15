# tQGA-XCD

## Overview
The code implements a tensorized Quantum Genetic Algorithm (tQGA) with Nelder–Mead local refinement for efficient inverse X-ray critical dimension (XCD) analysis.

The provided example demonstrates the reconstruction of a double-layer trapezoidal structure from X-ray scattering data.

## Contents

The repository mainly includes:

- X-ray scattering forward model
- Tensorized Quantum Genetic Algorithm (tQGA)
- Nelder–Mead local refinement
- Objective function and optimization procedures
- Example configuration and data for a double-layer trapezoidal structure

## Requirements

The required Python packages are listed in `requirements.txt`.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The code is implemented in Python and supports GPU acceleration through PyTorch when a compatible CUDA environment is available.

## Usage

All scripts whose filenames begin with `main` serve as the main entry points for generating the experimental results. The algorithm configurations in these scripts are consistent with those described in the paper. Run the corresponding script directly to generate the result data, which will be saved in the appropriate subfolder under the `data` directory.

- `main_DoubleTrape_QGANM_Seed.py`: Solves the double-layer trapezoidal structure using tQGA with a fixed random seed.
- `main_DoubleTrape_CMAES_Seed.py`: Solves the double-layer trapezoidal structure using CMA-ES with a fixed random seed.
- `main_DoubleTrape_DE_Seed.py`: Solves the double-layer trapezoidal structure using DE with a fixed random seed.
- `main_DoubleTrape_QGANM_RankAbla.py`: Performs an ablation study by replacing random subset sampling in the selective evolution mechanism with deterministic subset sampling based on objective-function ranking.

## License

Please refer to the `LICENSE` file for the license information.

## Reference

Liu S, Chen X, Liu S. Tensorized Quantum Genetic Algorithm With Selective Evolution Strategy for Thin‐Film Optical Inverse Problems[J]. Laser & Photonics Reviews, 2026, 20(11): e01880.  [DOI](https://doi.org/10.1002/lpor.202501880)
