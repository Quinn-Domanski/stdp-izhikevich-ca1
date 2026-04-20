import matplotlib.pyplot as plt
from brian2 import *
from model import IzhikevichNeuron

def run_baseline_validation():
    print("Running CA1 Baseline Validation Suite...")

    # ---------------------------------------------------------
    # TEST 1: Resting Potential (0 pA)
    # ---------------------------------------------------------
    neuron_rest = IzhikevichNeuron()
    neuron_rest.inject_current(0.0) 
    net_rest = Network(neuron_rest.group, neuron_rest.state_monitor, neuron_rest.spike_monitor)
    net_rest.run(100*ms)
    
    # ---------------------------------------------------------
    # TEST 2: Rheobase (Target: ~80 pA biological)
    # I=2.0 triggers exactly 1 spike, establishing our scale factor.
    # ---------------------------------------------------------
    neuron_rheo = IzhikevichNeuron()
    neuron_rheo.inject_current(2.0) 
    net_rheo = Network(neuron_rheo.group, neuron_rheo.state_monitor, neuron_rheo.spike_monitor)
    net_rheo.run(100*ms)

    # ---------------------------------------------------------
    # TEST 3: Active Firing (Target: ~200 pA biological)
    # I=5.0 triggers multiple spikes to show healthy recovery.
    # ---------------------------------------------------------
    neuron_active = IzhikevichNeuron()
    neuron_active.inject_current(5.0) 
    net_active = Network(neuron_active.group, neuron_active.state_monitor, neuron_active.spike_monitor)
    net_active.run(300*ms)

    # ---------------------------------------------------------
    # Plotting & Saving the Validation Figure
    # ---------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot Test 1
    ax1.plot(neuron_rest.state_monitor.t/ms, neuron_rest.state_monitor.v[0], color='blue', linewidth=2)
    ax1.set_title('Test 1: Resting Potential ($I=0.0$)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.axhline(-66.5, color='red', linestyle='--', alpha=0.5, label='Target Rest: -66.5 mV')
    ax1.set_ylim(-75, -55)
    ax1.legend()

    # Plot Test 2
    ax2.plot(neuron_rheo.state_monitor.t/ms, neuron_rheo.state_monitor.v[0], color='green', linewidth=2)
    ax2.set_title('Test 2: Rheobase Threshold ($I=2.0$, Scales to ~80 pA)')
    ax2.set_ylabel('Voltage (mV)')
    
    # Plot Test 3
    ax3.plot(neuron_active.state_monitor.t/ms, neuron_active.state_monitor.v[0], color='purple', linewidth=2)
    ax3.set_title('Test 3: Active Regular Spiking ($I=5.0$, Scales to ~200 pA)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Voltage (mV)')
    
    plt.tight_layout()
    plt.savefig("CA1_Complete_Validation.png", dpi=300, bbox_inches='tight')
    print("Success! Saved as 'CA1_Complete_Validation.png'")

if __name__ == '__main__':
    run_baseline_validation()