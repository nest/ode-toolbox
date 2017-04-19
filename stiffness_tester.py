import pygsl.odeiv as odeiv
import numpy as np
import Numeric
import math


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

        self.state = np.asarray([self.E_L, 0.0, 0.0, 0.0, 0.0])

    def get_initial_state(self):
        return [self.E_L, 0.0, 0.0, 0.0, 0.0]

    def set_state(self, state):
        self.state = state

    def initial_values(self):
        return np.asarray([0, math.e / self.tau_synE, 0, -math.e / self.tau_synI, 0])
        
    @staticmethod
    def step(t, y, params):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        f = Numeric.zeros((real_this.dimension,), Numeric.Float)

        I_syn_exc = y[2] * (y[0] - real_this.E_ex)
        I_syn_inh = y[4] * (y[0] - real_this.E_in)
        I_leak = real_this.g_L * (y[0] - real_this.E_L)

        f[0] = (-I_leak - I_syn_exc - I_syn_inh + real_this.I_e) / real_this.C_m

        f[1] = -y[1] / real_this.tau_synE
        f[2] = y[1] - (y[2] / real_this.tau_synE)

        f[3] = -y[3] / real_this.tau_synI
        f[4] = y[3] - (y[4] / real_this.tau_synI)
        
        return f


    @staticmethod
    def jac(t, y, t_m):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        dfdy = Numeric.zeros((real_this.dimension,real_this.dimension), Numeric. Float)
        dfdy[0, 0] = (real_this.g_L- y[2]-y[4])/real_this.C_m
        dfdy[0, 1] = 0
        dfdy[0, 2] = -y[0]/real_this.C_m
        dfdy[0, 3] = 0
        dfdy[0, 4] = y[0]/real_this.C_m
        dfdy[1, 0] = 0
        dfdy[1, 1] = -1/real_this.tau_synE
        dfdy[1, 2] = 0
        dfdy[1, 3] = 0
        dfdy[1, 4] = 0
        dfdy[2, 0] = 0
        dfdy[2, 1] = 1
        dfdy[2, 2] = -1/real_this.tau_synE
        dfdy[2, 3] = 0
        dfdy[2, 4] = 0
        dfdy[3, 0] = 0       
        dfdy[3, 1] = 0
        dfdy[3, 2] = 0
        dfdy[3, 3] = -1/real_this.tau_synI
        dfdy[3, 4] = 0
        dfdy[4, 0] = 0
        dfdy[4, 1] = 0
        dfdy[4, 2] = 0
        dfdy[4, 3] = 1
        dfdy[4, 4] = -1/real_this.tau_synI
        dfdt = Numeric.zeros((real_this.dimension,))
        return dfdy, dfdt

    @staticmethod
    def threshold(V_m):
        real_this = iaf_cond_alpha("iaf_cond_alpha_real_this")
        if V_m >real_this.V_th:
            return True
        else:
            return False


class hypothetical_stiff_neuron:
    def __init__(self, name):
        self.name = name
        self.dimension = 2
       
    def get_initial_state(self): 
        return (1,1)

    def initial_values(self):
        return [0, 0]        

    @staticmethod
    def step(t, y, params):
       
        real_this = hypothetical_stiff_neuron("hypothetical_stiff_neuron")
        f = Numeric.zeros((real_this.dimension,), Numeric.Float)
        f[0] = -100 * y[0]
        f[1] = -2 * y[1] + y[0]

        #print "    " + str(t)
        return f

    @staticmethod
    def jac(t, y, t_m):
        real_this = hypothetical_stiff_neuron("hypothetical_stiff_neuron")
        dfdy = Numeric.zeros((real_this.dimension,real_this.dimension), Numeric.Float) 
        dfdy[0, 0] = -100
        dfdy[0, 1] = 0.0
        dfdy[1, 0] = 1.0
        dfdy[1, 1] = -2.0
        dfdt = Numeric.zeros((real_this.dimension,))
        return dfdy, dfdt

class aeif_cond_alpha:


    def __init__(self, name):
        self.name = name
        self.dimension = 6
        self.tau_synE = 0.2
        self.tau_synI = 2.0
        self.E_ex = 0.0
        self.E_in = -85.0
        self.g_L = 30.0
        self.I_e = 0.0
        self.E_L = -70.6
        self.C_m = 281.0
        self.V_th = -50.4
        self.V_peak = 0.0
        self.tau_w = 144.0
        self.a = 4.0
        self.Delta_T = 2.0 #KLEINES DELTA GIBT PROBLEME FUER DAS IMPLIZITE VERFAHREN. JAKOBI-MATRIX?

        self.state = np.asarray([self.E_L, 0.0, 0.0, 0.0, 0.0, 0.0])

    def get_initial_state(self):
        return [self.E_L, 0.0, 0.0, 0.0, 0.0, 0.0]

    def set_state(self, state):
        self.state = state

    def initial_values(self):
        return np.asarray([0, math.e / self.tau_synE, 0, -math.e / self.tau_synI, 0, 0])
        #return np.asarray([0, 0, 0, 0, 0, 0])
        
    @staticmethod
    def step(t, y, params):
        real_this = aeif_cond_alpha("aeif_cond_alpha_real_this")
        f = Numeric.zeros((real_this.dimension,), Numeric.Float)

        I_syn_exc = y[2] * (y[0] - real_this.E_ex) # e-Funktion: tauE= 0.01; y[0]-E_ex positiv, aber nicht so gross
        I_syn_inh = y[4] * (y[0] - real_this.E_in)
        I_spike = real_this.g_L * real_this.Delta_T * math.exp((y[0] - real_this.V_th) / real_this.Delta_T)

        f[0] = (real_this.g_L * (y[0] - real_this.E_L) + I_spike - I_syn_exc - I_syn_inh - y[5] + real_this.I_e) / real_this.C_m

        f[1] = -y[1] / real_this.tau_synE
        f[2] = y[1] - (y[2] / real_this.tau_synE)

        f[3] = -y[3] / real_this.tau_synI
        f[4] = y[3] - (y[4] / real_this.tau_synI)
        f[5] = (real_this.a * (y[0] - real_this.E_L) - y[5]) / real_this.tau_w

        return f


    @staticmethod
    def jac(t, y, t_m):
        real_this = aeif_cond_alpha("aeif_cond_alpha_real_this")
        dfdy = Numeric.zeros((real_this.dimension,real_this.dimension), Numeric. Float)

        dfdy[0, 0] = (real_this.g_L + real_this.g_L * math.exp((y[0]-real_this.V_th)/real_this.Delta_T) - y[2] -y[4])/real_this.C_m
        dfdy[0, 1] = 0
        dfdy[0, 2] = -y[0]/real_this.C_m
        dfdy[0, 3] = 0
        dfdy[0, 4] = -y[0]/real_this.C_m
        dfdy[0, 5] = -1/real_this.C_m
        dfdy[1, 0] = 0
        dfdy[1, 1] = -1/real_this.tau_synE
        dfdy[1, 2] = 0
        dfdy[1, 3] = 0
        dfdy[1, 4] = 0
        dfdy[1, 5] = 0
        dfdy[2, 0] = 0
        dfdy[2, 1] = 1
        dfdy[2, 2] = -1/real_this.tau_synE
        dfdy[2, 3] = 0
        dfdy[2, 4] = 0
        dfdy[2, 5] = 0
        dfdy[3, 0] = 0       
        dfdy[3, 1] = 0
        dfdy[3, 2] = 0
        dfdy[3, 3] = -1/real_this.tau_synI
        dfdy[3, 4] = 0
        dfdy[3, 5] = 0
        dfdy[4, 0] = 0
        dfdy[4, 1] = 0
        dfdy[4, 2] = 0
        dfdy[4, 3] = 1
        dfdy[4, 4] = -1/real_this.tau_synI
        dfdy[4, 5] = 0
        dfdy[5, 0] = real_this.a/real_this.tau_w
        dfdy[5, 1] = 0
        dfdy[5, 2] = 0
        dfdy[5, 3] = 0
        dfdy[5, 4] = 0
        dfdy[5, 5] = -1/real_this.tau_w
        dfdt = Numeric.zeros((real_this.dimension,))
        return dfdy, dfdt


class aeif_cond_exp:

    def __init__(self, name):
        self.name = name
        self.dimension = 4
        self.tau_synE = 0.1
        self.tau_synI = 2.0
        self.E_ex = 0.0
        self.E_in = -85.0
        self.g_L = 30.0
        self.I_e = 0.0
        self.E_L = -70.6
        self.C_m = 281.0
        self.V_th = -50.4
        self.V_peak = 0.0
        self.tau_w = 144.0
        self.a = 4.0
        self.Delta_T = 2.0 #KLEINES DELTA GIBT PROBLEME FUER DAS IMPLIZITE VERFAHREN. JAKOBI-MATRIX?


        self.state = np.asarray([self.E_L, 0.0, 0.0, 0.0])

    def get_initial_state(self):
        return [self.E_L, 0.0, 0.0, 0.0]

    def set_state(self, state):
        self.state = state

    def initial_values(self):
        return np.asarray([0, 1, 1, 0])
        #return np.asarray([0, 0, 0, 0])
        
    @staticmethod
    def step(t, y, params):
        real_this = aeif_cond_exp("aeif_cond_exp_real_this")
        f = Numeric.zeros((real_this.dimension,), Numeric.Float)

        I_syn_exc = y[1] * (y[0] - real_this.E_ex) # e-Funktion: tauE= 0.01; y[0]-E_ex positiv, aber nicht so gross
        I_syn_inh = y[2] * (y[0] - real_this.E_in)
        I_spike = real_this.g_L * real_this.Delta_T * math.exp((y[0] - real_this.V_th) / real_this.Delta_T)

        f[0] = (real_this.g_L * (y[0] - real_this.E_L) + I_spike - I_syn_exc - I_syn_inh - y[3] + real_this.I_e) / real_this.C_m

        f[1] = -y[1] / real_this.tau_synE

        f[2] = -y[2] / real_this.tau_synI

        f[3] = (real_this.a * (y[0] - real_this.E_L) - y[3]) / real_this.tau_w

        return f


    @staticmethod
    def jac(t, y, t_m):
        real_this = aeif_cond_exp("aeif_cond_exp_real_this")
        dfdy = Numeric.zeros((real_this.dimension,real_this.dimension), Numeric. Float)

        dfdy[0, 0] = (real_this.g_L + real_this.g_L * math.exp((y[0]-real_this.V_th)/real_this.Delta_T) - y[1] -y[2])/real_this.C_m
        dfdy[0, 1] = -y[0]/real_this.C_m
        dfdy[0, 2] = -y[0]/real_this.C_m
        dfdy[0, 3] = -1/real_this.C_m
        dfdy[1, 0] = 0
        dfdy[1, 1] = -1/real_this.tau_synE
        dfdy[1, 2] = 0
        dfdy[1, 3] = 0
        dfdy[2, 0] = 0
        dfdy[2, 1] = 0
        dfdy[2, 2] = -1/real_this.tau_synE
        dfdy[2, 3] = 0
        dfdy[3, 0] = real_this.a/real_this.tau_w      
        dfdy[3, 1] = 0
        dfdy[3, 2] = 0
        dfdy[3, 3] = -1/real_this.tau_w
        dfdt = Numeric.zeros((real_this.dimension,))
        return dfdy, dfdt

    @staticmethod
    def threshold(V_m):
        real_this = aeif_cond_exp("aeif_cond_exp_real_this")
        if V_m > real_this.V_th:
            return True
        else:
            return False


def make_stiffness_test_for(neuron):
    print("Runs stiffness test for the neuron " + neuron.name)
    h = 0.6
    sim_time = 2.0
    simulation_slots = int(round(sim_time / h))
    print "Simulation slots:"+ str(simulation_slots) 

    gen_inh = generate_spike_train(simulation_slots)
   
    
    print ("######### rk imp #########")
    step_min_imp, step_average_imp = evaluate_integrator(h, 
                                       sim_time,
                                       simulation_slots,
                                       odeiv.step_bsimp,
                                       neuron.step,
                                       neuron.jac,
                                       [gen_inh] * neuron.dimension,
                                       neuron.get_initial_state(),
                                       neuron.initial_values(),
                                       None)
    print ("######### rk expl ########")
    step_min_exp, step_average_exp = evaluate_integrator(h,
                                       sim_time,
                                       simulation_slots,
                                       odeiv.step_rk4,
                                       neuron.step,
                                       neuron.jac,
                                       [gen_inh] * neuron.dimension,
                                       neuron.get_initial_state(),
                                       neuron.initial_values(),
                                       None)
    print ("######### results #######")
    print "min_imp: {} min_exp: {}".format(step_min_imp, step_min_exp)
    print "avg_imp: {} avg_exp: {}".format(step_average_imp, step_average_exp)
    print ("########## end ##########")


def evaluate_integrator(h,
                        sim_time,
                        simulation_slots,
                        integrator,
                        step_function,
                        jacobian,
                        spikes,
                        y,
                        initial_values,
                        threshold):
    s_min = h  # the minimal step size cannot be larger than the maximal stepsize h

    step = integrator(len(y), step_function, jacobian)
    control = odeiv.control_y_new(step, 1e-2, 1e-2) 
    evolve = odeiv.evolve(step, control, len(y))

    t=0.0
    step_counter=0
    for time_slot in range(simulation_slots):
        #t = 0.0
        t_new=t+h
        print "Start while loop at slot " + str(time_slot)
        while t < t_new:
            t_old = t
            t, h_, y = evolve.apply(t, t_new, h, y) # h_ is NOT the reached step size but the suggested next step size!
            step_counter+=1
            s_min_old = s_min
            s_min = min(s_min, t-t_old)
            print str(time_slot) + ":   t=%f, current stepsize=%f y=" % (t, t-t_old ), y
        s_min= s_min_old
        print "End while loop"
        
        if threshold is not None and threshold(y[0]):
            y[0] = -70.
            print("crossed")

        for idx, initial_value in enumerate(initial_values):
            y[idx] += initial_value * spikes[idx][time_slot]
    step_average= sim_time/step_counter
    return s_min_old, step_average


def generate_spike_train(SIMULATION_SLOTS):
    spike_train = [0.0, 2.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    return spike_train * int(math.ceil(float(SIMULATION_SLOTS)/len(spike_train)))

if __name__ == "__main__":
    make_stiffness_test_for(iaf_cond_alpha("iaf_cond_alpha"))
    make_stiffness_test_for(hypothetical_stiff_neuron("hypothetical_stiff_neuron"))
    make_stiffness_test_for(aeif_cond_alpha("aeif_cond_alpha"))
    make_stiffness_test_for(aeif_cond_exp("aeif_cond_exp"))
