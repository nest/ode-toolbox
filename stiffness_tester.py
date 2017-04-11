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

    def get_initial_state(self): 
        return (self.E_L, 0.0, 0.0, 0.0, 0.0)

    def initial_values(self):
        return [0, 0, math.e / self.tau_synE, 0, -math.e / self.tau_synI]
        
    @staticmethod
    def step(t, y, params):
       
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        f = Numeric.zeros((real_this.dimension,), Numeric.Float)

        I_syn_exc = y[1] * (y[0] - real_this.E_ex)
        I_syn_inh = y[3] * (y[0] - real_this.E_in)
        I_leak = real_this.g_L * (y[0] - real_this.E_L)

        f[0] = (-I_leak - I_syn_exc - I_syn_inh + real_this.I_e) / real_this.C_m

        f[1] = -y[2] / real_this.tau_synE
        f[2] = y[2] - (y[1] / real_this.tau_synE)

        f[3] = -y[4] / real_this.tau_synI
        f[4] = y[4] - (y[3] / real_this.tau_synI)
        #print "    " + str(t)
        return f

    @staticmethod
    def jac(t, y, t_m):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        dfdy = Numeric.zeros((5,5), Numeric. Float)
        dfdy[0, 0] = real_this.g_L- y[1]
        dfdy[0, 1] = -y[0]
        dfdy[0, 2] = 0
        dfdy[0, 3] = -y[0]
        dfdy[0, 4] = 0
        dfdy[1, 0] = 0
        dfdy[1, 1] = 0
        dfdy[1, 2] = -1/real_this.tau_synE
        dfdy[1, 3] = 0
        dfdy[1, 4] = 0
        dfdy[2, 0] = 0
        dfdy[2, 1] = -1/real_this.tau_synE
        dfdy[2, 2] = 1
        dfdy[2, 3] = 0
        dfdy[2, 4] = 0
        dfdy[3, 0] = 0       
        dfdy[3, 1] = 0
        dfdy[3, 2] = 0
        dfdy[3, 3] = 0
        dfdy[3, 4] = -1/real_this.tau_synI
        dfdy[4, 0] = 0
        dfdy[4, 1] = 0
        dfdy[4, 2] = 0
        dfdy[4, 3] = -1/real_this.tau_synI
        dfdy[4, 4] = 1
        dfdt = Numeric.zeros((5,))
        return dfdy, dfdt

    def threshold(V_m):
      if V_m > self.V_th:
          return True
      else:
          return False

class stiff_ODE:
    def __init__(self, name):
        self.name = name
        self.dimension = 2
        self.tau_synE = 0.2
        self.tau_synI = 2.0
        self.E_ex = 0.0
        self.E_in = -85.0
        self.g_L = 16.6667
        self.I_e = 0.0
        self.E_L = -70.0
        self.C_m = 250
        self.V_th = -55.0

    def get_initial_state(self): 
        return (1,1)

    def initial_values(self):
        return [0, 0]        

    @staticmethod
    def step(t, y, params):
       
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        f = Numeric.zeros((real_this.dimension,), Numeric.Float)
        f[0] = -100 * y[0]
        f[1] = -2 * y[1] + y[0]

        #print "    " + str(t)
        return f
    @staticmethod
    def jac(t, y, t_m):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        dfdy = Numeric.zeros((2,2), Numeric.Float) 
        dfdy[0, 0] = -100
        dfdy[0, 1] = 0.0
        dfdy[1, 0] = 1.0
        dfdy[1, 1] = -2.0
        dfdt = Numeric.zeros((2,))
        return dfdy, dfdt

def make_stiffness_test_for(neuron):
    print("Runs stiffness test for the neuron " + neuron.name)
    h = 0.1
    sim_time = 0.5
    simulation_slots = int(round(sim_time / h))
    print "Simulation slots:"+ str(simulation_slots) 

    gen_inh = generate_spike_train(simulation_slots)
    print ("######### rk imp #########")
    step_min_imp = evaluate_integrator(h, 
                                       sim_time,
                                       simulation_slots,
                                       odeiv.step_bsimp,
                                       iaf_cond_alpha.step,
                                       iaf_cond_alpha.jac,
                                       [gen_inh] * neuron.dimension,
                                       neuron.get_initial_state(),
                                       neuron.initial_values()
                                       None)
    print ("######### rk expl ########")
    step_min_exp = evaluate_integrator(h,
                                       sim_time,
                                       simulation_slots,
                                       odeiv.step_rk4,
                                       iaf_cond_alpha.step,
                                       iaf_cond_alpha.jac,
                                       [gen_inh] * neuron.dimension,
                                       neuron.get_initial_state(),
                                       neuron.initial_values(),
                                       None)
    print ("######### results #######")
    print "imp: {} exp: {}".format(step_min_imp, step_min_exp)
    print ("########## end ##########")

def evaluate_integrator(h, sim_time, simulation_slots, integrator, step_function, jacobian, spikes, y, initial_values):
    s_min = h  # the minimal step size cannot be larger than the maximal stepsize h

    step = integrator(len(y), step_function, jacobian)
    control = odeiv.control_y_new(step, 1e-2, 1e-2) 
    evolve = odeiv.evolve(step, control, len(y))

    for time_slot in range(simulation_slots):
        t = 0.0
        print "Start while loop at slot " + str(time_slot)
        while t < h:
            t_old = t
            t, h_, y = evolve.apply(t, h, 0.1, y) # h_ is NOT the reached step size but the suggested next step size?
            s_min = min(s_min, t-t_old)
            print str(time_slot) + ":   t=%f, current stepsize=%f y=" % (t, s_min ), y
        print "End while loop"
        
        if threshold is not None and threshold(y[0]):
            print("crossed")        
        for idx, initial_value in enumerate(initial_values):
            y[idx] += initial_value * spikes[idx][time_slot]
    return s_min

def generate_spike_train(SIMULATION_SLOTS):
    spike_train = [0.0, 0.1,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    return spike_train * max(int(SIMULATION_SLOTS/len(spike_train)), 1)

if __name__ == "__main__":
    make_stiffness_test_for(iaf_cond_alpha("iaf_cond_alpha"))
    make_stiffness_test_for(iaf_cond_alpha("stiff_ODE"))
