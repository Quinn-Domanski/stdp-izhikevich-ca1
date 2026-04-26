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
        
        self.group.tau_g = 5*ms # time for neurotransmitters to clear
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
    def __init__(self, pre_group, post_group, A_pre=0.1, A_post=-0.105, w_max=10.0):
        """
        STDP synapse that triggers conductance (g) instead of raw voltage
        """
        self.syn_eqs = '''
        w : 1
        taupre : second
        taupost : second
        w_max : 1
        w_min : 1
        Apre : 1   
        Apost : 1  
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        '''
        
        self.on_pre_eqs = '''
        g_post += w                
        apre += Apre               
        w = clip(w + apost, w_min, w_max) 
        '''
        
        self.on_post_eqs = '''
        apost += Apost             
        w = clip(w + apre, w_min, w_max)  
        '''
        
        self.synapses = Synapses(pre_group, post_group, model=self.syn_eqs, 
                                on_pre=self.on_pre_eqs, on_post=self.on_post_eqs)
                                
        # Temporarily store our Python arguments so we can use them later
        self._Apre = A_pre
        self._Apost = A_post
        self._wmax = w_max

    def connect(self, i, j, start_weight=2.0):
        """Helper function to connect indices and set starting weight"""
        self.synapses.connect(i=i, j=j)
        
        self.synapses.taupre = 20*ms
        self.synapses.taupost = 20*ms
        self.synapses.w_max = self._wmax
        self.synapses.w_min = 0.0
        self.synapses.Apre = self._Apre
        self.synapses.Apost = self._Apost
        self.synapses.w = start_weight
    