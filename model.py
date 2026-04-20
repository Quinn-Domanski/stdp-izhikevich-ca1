from brian2 import *

class IzhikevichNeuron:
    # ---------------------------------------------------------
    # BIOLOGICAL CA1 BASELINE PARAMETERS
    # c = -66.5 (Resting potential from Campanac & Debanne, 2008)
    # b = 0.2347 (Algebraically tuned to prevent baseline drift at c=-66.5)
    # a = 0.02, d = 8.0 (Standard Regular Spiking recovery dynamics)
    # ---------------------------------------------------------
    def __init__(self, a=0.02, b=0.2347, c=-66.5, d=8.0, N=1):
        """
        Baseline Izhikevich model representing a rat CA1 Pyramidal Neuron.
        N: Number of neurons in the group
        """
        
        # Izhikevich differential equations
        self.eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
        du/dt = (a*(b*v - u)) / ms : 1
        I : 1  # Injected dc-current (Dimensionless. 1 unit = ~40 pA)
        a : 1
        b : 1
        c : 1
        d : 1
        '''
        
        # Spike threshold and reset logic
        self.threshold_condition = 'v >= 30'
        self.reset_equations = '''
        v = c
        u = u + d
        '''
        
        # Build Brian2 NeuronGroup using Euler integration
        self.group = NeuronGroup(
            N, 
            model=self.eqs, 
            threshold=self.threshold_condition, 
            reset=self.reset_equations, 
            method='euler'
        )
        
        # Initialize standard parameters
        self.group.a = a
        self.group.b = b
        self.group.c = c
        self.group.d = d
        
        # Set exact resting state to prevent initial drift
        self.group.v = c
        self.group.u = b * c
        self.group.I = 0.0  
        
        # Attach monitors to record the virtual patch-clamp data
        self.state_monitor = StateMonitor(self.group, ['v', 'u'], record=True)
        self.spike_monitor = SpikeMonitor(self.group)

    def inject_current(self, current_value):
        """Injects a steady dimensionless current to trigger spiking."""
        self.group.I = current_value