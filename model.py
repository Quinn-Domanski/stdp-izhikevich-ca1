from brian2 import *

class IzhikevichNeuron:
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, N=1):
        """
        Baseline Izhikevich model.
        N: Number of neurons to create in this group.
        """
        
        self.eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
        du/dt = (a*(b*v - u)) / ms : 1
        I : 1  
        a : 1
        b : 1
        c : 1
        d : 1
        '''
        
        self.threshold_condition = 'v >= 30'
        self.reset_equations = '''
        v = c
        u = u + d
        '''
        
        # We now use the 'N' variable here so the group can be any size
        self.group = NeuronGroup(
            N, 
            model=self.eqs, 
            threshold=self.threshold_condition, 
            reset=self.reset_equations, 
            method='euler'
        )
        
        self.group.a = a
        self.group.b = b
        self.group.c = c
        self.group.d = d
        
        # Set initial resting state variables
        # Note: Using -70.0 (the math equilibrium) instead of 'c' 
        # prevents that initial "jump" you saw in earlier tests.
        self.group.v = -70.0
        self.group.u = b * -70.0
        self.group.I = 0.0  
        
        self.state_monitor = StateMonitor(self.group, ['v', 'u'], record=True)
        self.spike_monitor = SpikeMonitor(self.group)

    def inject_current(self, current_value):
        """Inject current to all neurons in the group"""
        self.group.I = current_value

class IzhikevichNeuronConductanceSynapse:
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, N=1):
        """
        Baseline Izhikevich model upgraded to support conductance-based synapse model.
        N: Number of neurons to create in this group.
        """
        
        self.eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I_ext + I_syn) / ms : 1
        du/dt = (a*(b*v - u)) / ms : 1

        I_syn = g * (E_eq - v) : 1  # current coming from synapses connected driven by internal voltage difference
        dg/dt = -g / tau_g : 1  # rate of neurotransmitters clearing and chemical gate closing

        I_ext : 1  # injected current  
        a : 1
        b : 1
        c : 1
        d : 1
        tau_g : second
        E_eq : 1
        '''
        
        self.threshold_condition = 'v >= 30'
        self.reset_equations = '''
        v = c
        u = u + d
        '''
        
        # We now use the 'N' variable here so the group can be any size
        self.group = NeuronGroup(
            N, 
            model=self.eqs, 
            threshold=self.threshold_condition, 
            reset=self.reset_equations, 
            method='euler'
        )
        
        self.group.a = a
        self.group.b = b
        self.group.c = c
        self.group.d = d
        
        self.group.tau_g = 5*ms 
        self.group.E_eq = 0.0
        # Set initial resting state variables
        # Note: Using -70.0 (the math equilibrium) instead of 'c' 
        # prevents that initial "jump" you saw in earlier tests.
        self.group.v = -70.0
        self.group.u = b * -70.0
        self.group.I_ext = 0.0
        self.group.g = 0.0  
        
        self.state_monitor = StateMonitor(self.group, ['v', 'u', 'g', 'I_syn'], record=True)
        self.spike_monitor = SpikeMonitor(self.group)

    def inject_current(self, current_value):
        """Inject current to all neurons in the group"""
        self.group.I = current_value

class ConductanceSTDPSynapse:
    def __init__(self, pre_group, post_group, A_pre=2.0, A_post=-2.0, w_max=10.0):
        """
        STDP synapse that triggers conductance (g) instead of raw voltage
        """
        self.syn_eqs = '''
        w : 1
        tau_pre : second
        tau_post : second
        w_max : 1
        w_min : 1
        A_pre : 1
        A_post : 1
        dapre/dt = -apre/tau_pre : 1 (event-driven)
        dapost/dt = -apost/tau_post : 1 (event-driven)
        '''
        
        # spike opens doors (g_post) instead of just teleporting voltage
        self.on_pre_eqs = '''
        g_post += w                
        apre += A_pre              
        w = clip(w + apost, w_min, w_max) 
        '''
        
        self.on_post_eqs = '''
        apost += A_post            
        w = clip(w + apre, w_min, w_max)  
        '''
        
        self.synapses = Synapses(pre_group, post_group, model=self.syn_eqs, 
                                on_pre=self.on_pre_eqs, on_post=self.on_post_eqs)
                                
        # set default STDP constants internally so experiments stay clean
        self.synapses.tau_pre = 20*ms
        self.synapses.tau_post = 20*ms
        self.synapses.w_max = w_max
        self.synapses.w_min = 0.0
        self.synapses.A_pre = A_pre
        self.synapses.A_post = A_post

    def connect(self, i, j, start_weight=5.0):
        """Helper function to connect indices and set starting weight"""
        self.synapses.connect(i=i, j=j)
        self.synapses.w = start_weight


# class FluidIPNeuron:
#     def __init__(self, N=1, C=100.0, k_param=0.7, v_r=-70.0, v_t_base=-40.0, a=0.03, b=-2.0, c=-50.0, d=100.0):
#         """
#         The 2007 Generalized Izhikevich Model with Novel Fluid Intrinsic Plasticity.
#         All variables are dimensionless to match Phase 1 architecture perfectly.
#         """
#         self.eqs = '''
#         dv/dt = (k_param*(v - v_r)*(v - v_t) - u + I) / (C * ms) : 1
#         du/dt = (a*(b*(v - v_r) - u)) / ms : 1
        
#         # The Novelty: Fluid, asymptotically bounded E-S coupling
#         v_t = v_t_base - 5.0 * tanh(k_IP * pos_delta_w) : 1
        
#         pos_delta_w = clip(w_total - w_base, 0.0, 10.0) : 1
        
#         w_total : 1
#         I : 1
#         k_param : 1
#         v_r : 1
#         v_t_base : 1
#         a : 1
#         b : 1
#         c : 1
#         d : 1
#         C : 1
#         k_IP : 1
#         w_base : 1
#         '''
        
#         self.group = NeuronGroup(N, model=self.eqs, threshold='v >= 20.0', reset='v = c; u += d', method='euler')
        
#         # Initialize Parameters (Dimensionless)
#         self.group.k_param = k_param
#         self.group.v_r = v_r
#         self.group.v_t_base = v_t_base
#         self.group.a = a
#         self.group.b = b
#         self.group.c = c
#         self.group.d = d
#         self.group.C = C
#         self.group.k_IP = 0.5 
#         self.group.w_base = 5.0
        
#         # Initialize Variables to Perfect Equilibrium
#         self.group.v = v_r
#         self.group.u = 0.0  # Because v = v_r, b*(v-v_r) = 0, so u rests perfectly at 0
#         self.group.I = 0.0
#         self.group.w_total = 5.0
        
#         self.state_monitor = StateMonitor(self.group, ['v', 'v_t'], record=True)
#         self.spike_monitor = SpikeMonitor(self.group)
    