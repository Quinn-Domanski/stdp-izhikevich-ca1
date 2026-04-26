import matplotlib.pyplot as plt
from brian2 import *
from model import IzhikevichNeuronConductanceSynapse, ConductanceSTDPSynapse

def run_conductance_experiment():
    print("Running Conductance & STDP Biophysical Experiment...")
    start_scope()

    # 1. Build the Biological Hardware
    print("Initializing Conductance Neurons and STDP Synapse...")
    neurons = IzhikevichNeuronConductanceSynapse(N=2)
    
    # 2. Wire them together
    stdp_connection = ConductanceSTDPSynapse(neurons.group, neurons.group)
    # Connect Pre (0) to Post (1) with a low starting weight of 2.0
    stdp_connection.connect(i=0, j=1, start_weight=2.0)
    
    # 3. Create Monitors
    # The neuron's state_monitor is already tracking v, u, g, and I_syn!
    # We just need to add a monitor for the synaptic weight (w)
    syn_monitor = StateMonitor(stdp_connection.synapses, 'w', record=True)
    
    # Bundle everything into a Network
    net = Network(neurons.group, neurons.state_monitor, neurons.spike_monitor, 
                  stdp_connection.synapses, syn_monitor)

    # ---------------------------------------------------------
    # THE EXPERIMENT TIMELINE
    # ---------------------------------------------------------
    print("Letting network settle to resting potential...")
    net.run(30*ms)

    # PHASE 1: THE BASELINE EPSP
    print("Phase 1: Firing Pre-Neuron (Baseline EPSP)...")
    neurons.group.I_ext[0] = 50.0  # Shock Neuron 0
    net.run(2*ms)
    neurons.group.I_ext[0] = 0.0   # Turn shock off
    
    net.run(15*ms) # Wait and watch the EPSP curve

    # PHASE 2: THE LEARNING (LTP)
    print("Phase 2: Firing Post-Neuron (Triggering STDP LTP)...")
    neurons.group.I_ext[1] = 50.0  # Shock Neuron 1
    net.run(2*ms)
    neurons.group.I_ext[1] = 0.0 
    
    net.run(50*ms) # Let the network fully recover and the weight permanently update

    # PHASE 3: THE STRENGTHENED EPSP
    print("Phase 3: Firing Pre-Neuron again (Checking the newly learned weight)...")
    neurons.group.I_ext[0] = 50.0  
    net.run(2*ms)
    neurons.group.I_ext[0] = 0.0 
    
    net.run(40*ms) # Watch the new, massive EPSP curve

    # ---------------------------------------------------------
    # PLOTTING THE BIOPHYSICS
    # ---------------------------------------------------------
    print("Generating biological traces...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    time = neurons.state_monitor.t / ms
    
    # Top Graph: The Voltages
    ax1.plot(time, neurons.state_monitor.v[0], label='Pre-Synaptic (0)', color='black')
    ax1.plot(time, neurons.state_monitor.v[1], label='Post-Synaptic (1)', color='#1f77b4')
    ax1.set_title('Membrane Voltages: Notice the smooth EPSP curves!', fontweight='bold')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend(loc='upper right')
    
    # Middle Graph: The Chemical Doors (Conductance)
    # We plot g[1] because the doors are opening on the Post-synaptic cell
    ax2.plot(time, neurons.state_monitor.g[1], color='green', linewidth=2)
    ax2.set_title('Post-Synaptic Conductance ($g$)', fontweight='bold')
    ax2.set_ylabel('Open Channels ($g$)')
    
    # Bottom Graph: The STDP Memory (Weight)
    ax3.plot(syn_monitor.t / ms, syn_monitor.w[0], color='red', linewidth=3)
    ax3.set_title('Synaptic Weight ($w$)', fontweight='bold')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Connection Strength')
    
    plt.tight_layout()
    plt.savefig("Conductance_STDP_Proof.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    run_conductance_experiment()