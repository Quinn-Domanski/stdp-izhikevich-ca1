import matplotlib.pyplot as plt
from brian2 import *
from model import TwoCompartmentIzhikevichNeuronAMPA_NMDA, AMPA_NMDA_STDPSynapse

def run_ampa_nmda_stdp_experiment():
    print("Running AMPA/NMDA Two-Compartment STDP Experiment...")
    start_scope()

    # Build two neurons: neuron 0 = pre, neuron 1 = post
    neurons = TwoCompartmentIzhikevichNeuronAMPA_NMDA(N=2)

    # Connect neuron 0 to neuron 1 with AMPA/NMDA STDP synapse
    stdp_connection = AMPA_NMDA_STDPSynapse(neurons.group, neurons.group)
    stdp_connection.connect(i=0, j=1, start_weight=0.02)

    # Monitor synaptic weight
    syn_monitor = StateMonitor(stdp_connection.synapses, 'w', record=True)

    # Build explicit Brian2 network
    net = Network(
        neurons.group,
        neurons.state_monitor,
        neurons.spike_monitor,
        stdp_connection.synapses,
        syn_monitor
    )

    # Experiment parameters
    pre_current = 20.0
    post_current = 20.0
    phase3_current = 20.0
    pulse_duration = 2*ms

    # Let network settle
    net.run(30*ms)

    # Phase 1: baseline pre input
    print("Phase 1: Baseline pre-synaptic stimulation")
    neurons.group.I_ext[0] = pre_current
    net.run(pulse_duration)
    neurons.group.I_ext[0] = 0.0

    net.run(15*ms)

    # Phase 2: force post spike after pre spike for LTP
    print("Phase 2: Post-synaptic stimulation for STDP")
    neurons.group.I_ext[1] = post_current
    net.run(pulse_duration)
    neurons.group.I_ext[1] = 0.0

    net.run(100*ms)

    # Optional: make Phase 3 a pure readout by clearing traces.
    # Use this only if you do not want the test spike to further update w.
    # stdp_connection.synapses.apre = 0.0
    # stdp_connection.synapses.apost = 0.0

    # Phase 3: test learned synapse
    print("Phase 3: Re-test pre-synaptic input")
    neurons.group.I_ext[0] = phase3_current
    net.run(pulse_duration)
    neurons.group.I_ext[0] = 0.0

    net.run(60*ms)

    # Plotting
    time = neurons.state_monitor.t / ms

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Soma and dendrite voltage
    ax1.plot(time, neurons.state_monitor.v[0], label='Pre Soma Voltage')
    ax1.plot(time, neurons.state_monitor.v[1], label='Post Soma Voltage')
    ax1.plot(time, neurons.state_monitor.v_dend[1], label='Post Dendrite Voltage', linestyle='--')
    ax1.set_title('Soma and Dendritic Voltages')
    ax1.set_ylabel('Voltage-like value')
    ax1.legend(loc='upper right')

    # AMPA/NMDA conductance
    ax2.plot(time, neurons.state_monitor.g_ampa[1], label='AMPA conductance')
    ax2.plot(time, neurons.state_monitor.g_nmda[1], label='NMDA conductance')
    ax2.set_title('Post-Synaptic AMPA and NMDA Conductances')
    ax2.set_ylabel('Conductance')
    ax2.legend(loc='upper right')

    # AMPA/NMDA currents
    ax3.plot(time, neurons.state_monitor.I_ampa[1], label='AMPA current')
    ax3.plot(time, neurons.state_monitor.I_nmda[1], label='NMDA current')
    ax3.plot(time, neurons.state_monitor.I_syn[1], label='Total synaptic current', linestyle='--')
    ax3.set_title('Post-Synaptic Synaptic Currents')
    ax3.set_ylabel('Current-like value')
    ax3.legend(loc='upper right')

    # STDP weight
    ax4.plot(syn_monitor.t / ms, syn_monitor.w[0], label='Synaptic weight')
    ax4.set_title('STDP Synaptic Weight')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Weight')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_ampa_nmda_stdp_experiment()