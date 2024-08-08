# pybullet-helpers

![workflow](https://github.com/tomsilver/pybullet-helpers/actions/workflows/ci.yml/badge.svg)

Some utility functions for PyBullet. Copied and modified from [predicators](https://github.com/Learning-and-Intelligent-Systems/predicators), which in turn was heavily based on the pybullet-planning repository by Caelan Garrett (https://github.com/caelan/pybullet-planning/). In addition, the structure is loosely based off the pb_robot repository
by Rachel Holladay (https://github.com/rachelholladay/pb_robot). [Will Shen](https://shen.nz/) made huge contributions.

## Requirements

- Python 3.10+
- Tested on MacOS Catalina

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
