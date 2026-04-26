from brian2 import *

class IzhikevichNeuron:
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0):
        """
        baseline Izhikevich model with baseline "regular spiking" parameters as init function arguments
        """
        
        # Izhikevich differential equations
        # divide by ms to put onto ms time scale
        # ": 1" keeps variables dimensionless, such as u and v, as well as parameters a, b, c, d
        self.eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
        du/dt = (a*(b*v - u)) / ms : 1
        I : 1  # Injected dc-current
        a : 1
        b : 1
        c : 1
        d : 1
        '''
        
        # spike and reset logic
        # spiek threshold = 30, when reached reset internal voltage with c parameter
        # and increase negative feedback variable u
        self.threshold_condition = 'v >= 30'
        self.reset_equations = '''
        v = c
        u = u + d
        '''
        
        # build Brian2 Neurongroup with euler method to approximate trends
        self.group = NeuronGroup(
            1, 
            model=self.eqs, 
            threshold=self.threshold_condition, 
            reset=self.reset_equations, 
            method='euler'
        )
        
        # initialize parameters for this neuron
        self.group.a = a
        self.group.b = b
        self.group.c = c
        self.group.d = d
        
        # set initial resting state variables
        self.group.v = c
        self.group.u = b * c
        self.group.I = 0.0  # start with no injected current
        
        # attach monitors to record internal math at every ms to be graphed
        # SateMonitor acts as patc-clamp electrode that continuously saves current state v voltage and u negative feedback variable
        # SpikeMonitor logs when spike is fired
        self.state_monitor = StateMonitor(self.group, ['v', 'u'], record=True)
        self.spike_monitor = SpikeMonitor(self.group)

    def inject_current(self, current_value):
        """Inject current to trigger spiking"""
        self.group.I = current_value