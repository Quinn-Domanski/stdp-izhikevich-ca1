from brian2 import *
import matplotlib.pyplot as plt
from model import IzhikevichNeuron

def run_experiment():
    start_scope()

    # instantiate baseline regular spiking neuron
    print("Initializing baseline Izhikevich neuron...")
    neuron = IzhikevichNeuron()

    # create explicit network to tell brian2 which objects to simulate:
    # a group of Izhikevich modeled neurons, the internal state, and spikes
    net = Network(neuron.group, neuron.state_monitor, neuron.spike_monitor)

    # experimental timeline
    print("Running initial resting phase (50 ms)...")
    net.run(50*ms)

    # inject a step of dc-current I=10 as defined in the 2003 paper
    print("Injecting DC current step (I=10)...")
    neuron.inject_current(10.0)

    # run simulation for another 250ms to observe spikes
    print("Simulating active spiking phase (250 ms)...")
    net.run(250*ms)

    # plotting patch-clamp results
    print("Generating voltage trace plot...")
    plt.figure(figsize=(10, 4))
    
    # extract the simulated time and voltage from StateMonitor
    time_data = neuron.state_monitor.t / ms
    voltage_data = neuron.state_monitor.v[0]
    
    plt.plot(time_data, voltage_data, color='#1f77b4', linewidth=1.5)
    
    # formatting the graph
    plt.title('Izhikevich Model: Regular Spiking (RS) Baseline', fontsize=14, fontweight='bold')
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Membrane Potential (mV)', fontsize=12)
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Current Injection Start')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # display the plot window
    plt.show()

if __name__ == '__main__':
    run_experiment()