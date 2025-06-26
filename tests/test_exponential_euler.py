#
# test_exponential_euler.py
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

from matplotlib import pyplot as plt
import numpy as np
import pytest
import scipy

from tests.test_utils import _open_json

try:
    import pygsl
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False

import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator


class TestExponentialEuler:

    def generate_reference_lotka_volterra_timeseries(self):
        def lotka_volterra_ode(t, y, alpha, beta, delta, gmma):
            """
            Defines the Lotka-Volterra differential equations.

            Args:
                t: Time (not explicitly used in this autonomous system, but required by solve_ivp).
                y: Array of current population values [prey, predator].
                alpha: Prey natural growth rate.
                beta: Predation rate coefficient.
                delta: Predator growth efficiency coefficient.
                gmma: Predator natural death rate.

            Returns:
                Array of the derivatives [dx/dt, dy/dt].
            """
            prey, predator = y
            dprey_dt = alpha * prey - beta * prey * predator
            dpredator_dt = delta * prey * predator - gmma * predator
            return [dprey_dt, dpredator_dt]

        # 2. Set Parameters (Chosen to potentially exhibit stiffness)
        #    A large gmma might make the predator population crash quickly,
        #    introducing a faster time scale compared to prey recovery.
        alpha = 1.0  # Prey growth rate
        beta = 1.0   # Predation rate
        delta = 1.0  # Predator efficiency
        gmma = 3.0  # Predator death rate (relatively high)

        # 3. Set Initial Conditions and Time Span
        y0 = [10.0, 5.0]  # Initial populations [prey, predator]
        t_span = (0, 30)  # Time interval for integration (start, end)
        t_eval = np.linspace(t_span[0], t_span[1], 500) # Points where solution is stored

        # 4. Integrate the ODEs using a stiff solver
        #    'Radau' and 'BDF' are implicit methods suitable for stiff problems.
        #    'LSODA' can automatically detect stiffness and switch methods.
        #    We'll use 'Radau' explicitly here.
        sol = scipy.integrate.solve_ivp(
            lotka_volterra_ode,
            t_span,
            y0,
            method='Radau',  # Solver choice for stiff systems
            args=(alpha, beta, delta, gmma),
            t_eval=t_eval,  # Specify output times
            dense_output=True # Useful for smooth plotting if needed later
        )

        # 5. Check if the integration was successful
        if not sol.success:
            raise Exception(f"ODE integration failed: {sol.message}")

        # 6. Extract the results
        t = sol.t
        prey_pop = sol.y[0]
        predator_pop = sol.y[1]

        return t, prey_pop, predator_pop


    def test_exponential_euler(self):
        alpha = 1.0
        beta = 1.0
        delta = 1.0
        gmma = 3.0

        # Initial conditions from the previous example:
        x0 = 10.0
        y0 = 5.0

        indict = {"dynamics": [
        {
            "expression": "x' = alpha * x - beta * x * y", # Prey equation
            "initial_value": str(x0)                      # Initial prey population as string
        },
        {
            "expression": "y' = delta * x * y - gmma * y", # Predator equation
            "initial_value": str(y0)                      # Initial predator population as string
        }
        # Note: The third expression "V_m' = ..." from your example doesn't
        # belong to the standard Lotka-Volterra model, so it's omitted here.
    ],
    "parameters": {
        "alpha": str(alpha),    # Prey growth rate as string
        "beta": str(beta),      # Predation rate as string
        "delta": str(delta),    # Predator efficiency as string
        "gmma": str(gmma)     # Predator death rate as string
        # Note: Parameters "tau" and "E_L" from your example don't belong
        # to the standard Lotka-Volterra model, so they are omitted here.
    }
}
        T = 30
        h = 0.1
        x0 = 10.0
        y0 = 5.0

        result = odetoolbox.analysis(indict)
        assert len(result) == 1
        solver_dict = result[0]

        N = int(np.ceil(T / h) + 1)
        timevec = np.linspace(0., T, N)
        state = {sym: [] for sym in solver_dict["state_variables"]}
        state["timevec"] = []
        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": x0, "y": y0})
        analytic_integrator.reset()
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)
            state["timevec"].append(t)
            for sym, val in state_.items():
                state[sym].append(val)

        for k, v in state.items():
            state[k] = np.array(v)

        t, prey_pop, predator_pop = self.generate_reference_lotka_volterra_timeseries()



        fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes objects

        ax.plot(t, prey_pop, label='Prey Population (x)', color='blue')
        ax.plot(t, predator_pop, label='Predator Population (y)', color='red')

        ax.plot(state["timevec"], state["x"], label='Prey Population (x)', color='blue', linestyle="--")
        ax.plot(state["timevec"], state["y"], label='Predator Population (y)', color='red', linestyle="--")

        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.set_title(f'Lotka-Volterra Predator-Prey Model (Stiff Solver: Radau)\n'
                    f'α={alpha}, β={beta}, δ={delta}, γ={gmma}')

        # Add legend and grid
        ax.legend()
        ax.grid(True)

        # Set limits (optional, adjust as needed)
        ax.set_ylim(bottom=0) # Populations cannot be negative

        # Display the plot
        plt.show()

        # --- Optional: Plot the phase portrait (Predator vs Prey) ---
        fig_phase, ax_phase = plt.subplots(figsize=(7, 7))

        ax_phase.plot(prey_pop, predator_pop, color='purple')
        # Mark the start and end points
        ax_phase.plot(prey_pop[0], predator_pop[0], 'go', label='Start') # Green circle
        ax_phase.plot(prey_pop[-1], predator_pop[-1], 'mo', label='End') # Magenta circle


        ax_phase.set_xlabel('Prey Population (x)')
        ax_phase.set_ylabel('Predator Population (y)')
        ax_phase.set_title('Phase Portrait (Predator vs. Prey)')
        ax_phase.grid(True)
        ax_phase.legend()
        ax_phase.set_xlim(left=0)
        ax_phase.set_ylim(bottom=0)

        plt.show()


        # assert ......