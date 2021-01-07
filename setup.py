#!/usr/bin/env python
#
# setup.py
#
# This file is part of the NEST ODE toolbox.
#
# Copyright (C) 2017 The NEST Initiative
#
# The NEST ODE toolbox is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 2 of
# the License, or (at your option) any later version.
#
# The NEST ODE toolbox is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
#

from setuptools import setup

long_description = """Choosing the optimal solver for systems of ordinary differential equations (ODEs) is a critical step in dynamical systems simulation. ODE-toolbox is a Python package that assists in solver benchmarking, and recommends solvers on the basis of a set of user-configurable heuristics. For all dynamical equations that admit an analytic solution, ODE-toolbox generates propagator matrices that allow the solution to be calculated at machine precision. For all others, first-order update expressions are returned based on the Jacobian matrix.

In addition to continuous dynamics, discrete events can be used to model instantaneous changes in system state, such as a neuronal action potential. These can be generated by the system under test as well as applied as external stimuli, making ODE-toolbox particularly well-suited for applications in computational neuroscience."""

setup(name="odetoolbox",
      version="2.2",
      author="The NEST Initiative",
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering'],
      description="ODE-toolbox: Automatic selection and generation of integration schemes for systems of ordinary differential equations",
      keywords="computational neuroscience model ordinary differential equation ode dynamical dynamic simulation",
      license="GNU General Public License v2 (GPLv2)",
      long_description=long_description,
      packages=['odetoolbox'],
      package_dir={'odetoolbox': 'odetoolbox'},
      requires=['matplotlib', 'numpy', 'sympy'],
      scripts=['ode_analyzer.py'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      url="https://github.com/nest/ode-toolbox")
