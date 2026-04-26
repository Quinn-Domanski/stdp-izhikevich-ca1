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