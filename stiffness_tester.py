import pygsl.odeiv as odeiv
import Numeric
import math
import random

class iaf_cond_alpha:
    def __init__(self, name):
        self.name = name
        self.dimension = 5
        self.tau_synE = 0.2
        self.tau_synI = 2.0
        self.E_ex = 0.0
        self.E_in = -85.0
        self.g_L = 16.6667
        self.I_e = 0.0
        self.E_L = -70.0
        self.C_m = 250
        self.V_th = -55.0

    def get_state(self):
        return (self.E_L, 0.0, 0.0, 0.0, 0.0)

    def initial_values(self):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        return [0, 0, math.e / real_this.tau_synE, 0, -math.e / real_this.tau_synI]

    @staticmethod
    def step(t, y, params):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")

        I_syn_exc = y[1] * (y[0] - real_this.E_ex)
        I_syn_inh = y[3] * (y[0] - real_this.E_in)
        I_leak = real_this.g_L * (y[0] - real_this.E_L)

        f = Numeric.zeros((real_this.dimension,), Numeric.Float)
        f[0] = (-I_leak - I_syn_exc - I_syn_inh + real_this.I_e) / real_this.C_m

        f[1] = -y[2] / real_this.tau_synE
        f[2] = y[2] - (y[1] / real_this.tau_synE)

        f[3] = -y[4] / real_this.tau_synI
        f[4] = y[4] - (y[3] / real_this.tau_synI)
        print "    " + str(t)
        return f

def make_stiffness_test_for(neuron):
    print("Runs stiffness test for the neuron " + neuron.name)
    h = 0.1
    sim_time = 0.3
    simulation_slots = int(sim_time / h)

    gen_inh = generate_spike_train(simulation_slots)
    print ("######### rk imp #########")
    step_min_imp = evaluate_integrator(h,
                                       sim_time,
                                       simulation_slots,
                                       odeiv.step_rk4imp,
                                       iaf_cond_alpha.step,
                                       [gen_inh] * neuron.dimension,
                                       neuron.get_state(),
                                       neuron.initial_values())
    print ("######### rk expl ########")
    step_min_exp = evaluate_integrator(h,
                                       sim_time,
                                       simulation_slots,
                                       odeiv.step_rk4,
                                       iaf_cond_alpha.step,
                                       [gen_inh] * neuron.dimension,
                                       neuron.get_state(),
                                       neuron.initial_values())
    print ("######### results #######")
    print "imp: {} exp: {}".format(step_min_imp, step_min_exp)
    print ("########## end ##########")

def evaluate_integrator(h, sim_time, simulation_slots, integrator, step_function, spikes, y, initial_values):
    s_min = h  # the minimal step size cannot be larger than the maximal stepsize h

    step = integrator(len(y), step_function, None)
    control = odeiv.control_y_new(step, 1e-6, 1e-6) # vary absolute and relative error tolerance
    evolve = odeiv.evolve(step, control, len(y))

    for time_slot in range(simulation_slots):
        t = 0.0
        print "Start while loop at slot " + str(time_slot)
        while t < h:
            t, h_, y = evolve.apply(t, h, 0.05, y)
            print str(time_slot) + ":   t=%f, h_=%f y=" % (t, h_), y
            s_min = min(s_min, h_)
        print "End while loop"
        for idx, initial_value in enumerate(initial_values):
            y[idx] += initial_value * spikes[idx][time_slot]
    return s_min

def generate_spike_train(SIMULATION_SLOTS):
    spike_train = [0.0, 0.1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    return spike_train * max(int(SIMULATION_SLOTS/len(spike_train)), 1)

if __name__ == "__main__":
    make_stiffness_test_for(iaf_cond_alpha("iaf_cond_alpha"))
