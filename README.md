# OR Problems

A collection of operational research and optimisation problems implemented in Python, using [Gurobi](https://www.gurobi.com/) and [CVXPY](https://www.cvxpy.org/). Each problem comes with a worked solution and a workshop version that provides the data and plotting code but leaves the modelling to you.

## Problems

| Problem | Model class | Tools |
| --- | --- | --- |
| [Chebyshev centre](problems/chebyshev_centre) | LP | Gurobi |
| [Circuit board planning](problems/circuit_board_planning) | MIP | Gurobi, NetworkX |
| [Floor planning](problems/floor_planning) | MIP | Gurobi |
| [Geometric median (facility location)](problems/geometric_median) | SOCP | Gurobi |
| [Lyapunov function validation](problems/lyapunov_validation) | SDP | CVXPY |
| [Maximum cut](problems/max_cut) | MIP, SDP relaxation | Gurobi, CVXPY |
| [Minimum volume enclosing ellipsoid](problems/minimum_volume_ellipsoid) | Convex (log-det) | CVXPY, Gurobi |
| [Robust optimisation via sum of largest eigenvalues](problems/robust_optimisation) | SDP | CVXPY |
| [Separation problem (classification)](problems/separation_problem) | LP, SOCP | Gurobi |
| [Truss topology design](problems/truss_design) | SDP | CVXPY |

## Repository layout

Each folder in `problems/` contains:

- `solution.ipynb`: the full worked solution.
- `workshop.ipynb`: problem description, data generation and plotting, with the model left for you to build.
- `workshop.py`: the workshop notebook as a plain Python script, for those who prefer not to use Jupyter.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.13 or later.

```bash
git clone https://github.com/Mattmandav/OR_Problems.git
cd OR_Problems
uv sync
```

Then open any notebook in Jupyter or VS Code and select the `.venv` kernel.

The pip-installed `gurobipy` includes a size-limited licence, which is enough for most problems here. The circuit board planning solution needs a full licence; academics can get one free from [Gurobi's academic programme](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Contributing

Install the pre-commit hooks once:

```bash
uv run pre-commit install
```

On each commit these strip notebook outputs and keep every `workshop.py` in sync with its `workshop.ipynb`, so you can edit either one.
