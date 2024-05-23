# PyRoboPlan

[![PyRoboPlan Tests](https://github.com/sea-bass/pyroboplan/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sea-bass/pyroboplan/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/pyroboplan/badge/?version=latest)](https://pyroboplan.readthedocs.io/en/latest/?badge=latest)
![Coverage Status](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/sea-bass/e5e091166a18c68b26338793917d3bab/raw/pyroboplan-test-coverage.json)

Educational Python library for manipulator motion planning.

This library extensively uses the [Pinocchio](https://github.com/stack-of-tasks/pinocchio) Python bindings for modeling robot kinematics and dynamics.

For more information, refer to the [full documentation](https://pyroboplan.readthedocs.io/en/latest/).

By Sebastian Castro, 2024

![RRT based motion planning and trajectory execution](https://raw.githubusercontent.com/sea-bass/pyroboplan/main/docs/source/_static/gifs/pyroboplan_rrt_traj.gif)

![Cartesian motion planning](https://raw.githubusercontent.com/sea-bass/pyroboplan/main/docs/source/_static/gifs/pyroboplan_cartesian_path.gif)

---

## Setup

### From PyPi

```bash
pip3 install pyroboplan
```

### From Source

Clone this repository.

```bash
git clone https://github.com/sea-bass/pyroboplan.git
```

(Optional) Set up a virtual environment and install dependencies.

```bash
source scripts/setup_virtual_env.bash
```

Install this package and its dependencies.

```bash
pip3 install -e .
```
