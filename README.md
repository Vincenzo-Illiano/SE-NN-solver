# Neural solver for the 1D Schrödinger equation

This project uses PyTorch to approximate the ground-state wavefunction and energy of a particle in a one-dimensional potential. It solves the dimensionless eigenvalue problem

$$
\left[-\frac{d^2}{dx^2} + V(x)\right]\phi(x) = E\phi(x),
\qquad x\in[-A,A], \qquad \phi(-A)=\phi(A)=0.
$$

The model samples the potential on an equally spaced grid and passes its low Fourier modes through a feedforward network. An inverse FFT (pytorch default one) then reconstructs the predicted wavefunction. The model sets the endpoint values to zero, normalizes the wavefunction on the grid, and evaluates the energy using the Rayleigh quotient. Training uses randomly generated potentials. Evaluation compares predictions with finite-difference ground states. 

The general idea of this project is to provide a set of tools that can be used to study how well FFNNs perform at solving eigenvalue problems. For this reason, scripts are provided to create different types of datasets and models.  

## Setup

Use Python 3.10 or newer and PyTorch 2.6 or newer. 

```bash
python -m pip install "torch>=2.6"
```

The core workflow only needs PyTorch. The optional dataset plotting utility in `fno/utils/plotter.py` also uses Matplotlib and NumPy.

## Create data, train, and evaluate

In the example we use a small CPU run to verify the workflow.

### 1. Generate separate training and testing sets

```bash
python -m scripts.create_dataset -t -T --num-batches 4 --batch-size 8 -N 128 -A 1
```

With no explicit paths, `-t` writes to `data/training/trainingset.pt` and `-T` writes to `data/testing/testingset.pt`. Specify custom paths after either flag. Use `--settype` to choose `mixed`, `poly`, `smooth`, or `well`. Add `--gpu` to use CUDA when available or `--overwrite` to replace existing datasets. 

Training files store the generated potentials. The model's `A` defines their spatial coordinates during training, meaning that the potentials represent functions defined in the interval $[-A, A]$. The same applies to testing files. These, in addition to training files, also store the solution of the Schrödinger eq. generated using finite differences (for this reason they take a bit longer to generate).

Use the same number of grid points `N` for the model and both datasets, and the same `A` for the model and the testing set in order to properly test.

Using --help will provide additional information.

Note: GPU memory at the moment is not very well managed, especially when generating the testingset. For this reason, cuda might run out of memory when trying to generate too large testing sets.

### 2. Create a model

```bash
python -m scripts.create_model -m fno1d -p data/models/initial.pth \
    -N 128 -M 32 --hidden 128 128 -A 1
```

`-M` selects how many real FFT bins to retain, including the zero-frequency bin (so `-M 32` will use the first 32 modes). It must be between 1 and `N // 2 + 1`. The values after `--hidden` set the widths of the hidden layers.

Using --help will provide additional information.

### 3. Train

```bash
python -m scripts.train -m fno1d -p data/models/initial.pth \
    -t data/training/trainingset.pt -s data/models/trained.pth \
    -e 5 -l minimumE_loss -d cpu --shuffleset
```

The training script uses Adam with a learning rate of `1e-4`. `minimumE_loss` minimizes the mean energy in order to target the ground state. `minimumErr_loss` minimizes the squared eigenvalue residual on the interior grid points. Excited eigenstates can also have a small residual, so to target the groundstate it's optimal to use `minimumE_loss` first and later move on to `minimumErr_loss`.

The `-s` option sets the output checkpoint path. If omitted, training overwrites the input checkpoint. The default training device is CUDA. Use `-d cpu` for a CPU run.

Using --help will provide additional information.

### 4. Evaluate

```bash
python -m scripts.test -p data/models/trained.pth -T data/testing/testingset.pt -d cpu
```

The script prints the mean test loss over all batches. The loss is the wavefunction MSE after choosing the overall sign that best aligns each reference with its prediction (since wavefunctions with different sign are equivalent), plus the energy MSE multiplied by `--energy-weight` (default 1, use 0 if you don't care whether your model actually finds the groundstate). Testing uses CUDA when available and otherwise runs on the CPU. 

Using --help will provide additional information.

### 5. Plot a prediction

```bash
python -m pip install matplotlib numpy
scripts/plot_solution -T data/testingset.pt -p data/models/trained.pth -i 0 0
```

The plot compares the model and reference ground states for one potential. Use `-i BATCH SAMPLE` to select a zero-based batch and sample. The default is `0 0`.

## Source layout

| Path | Purpose |
| --- | --- |
| `scripts/` | Command-line entry points for dataset generation, model creation, training, and evaluation |
| `fno/models/fno1d.py` | Fourier-based 1d network and checkpoint loading |
| `fno/datasets/generators.py` | Random potential generation and dataset serialization |
| `fno/training.py`, `fno/losses.py` | Training loop and physics-based losses |
| `fno/testing.py` | Batched evaluation and sign-aligned test loss |
| `fno/utils/physics_helper.py` | Finite-difference Hamiltonian and reference eigensolver |
