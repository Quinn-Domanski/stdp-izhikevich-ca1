import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import *
from model import IzhikevichNeuron

prefs.codegen.target = "numpy" 

def run_baseline_validation():
    print("\n--- Running Phase 1: Baseline Validation ---")
    
    # --- Test 1: Stable Rest ---
    start_scope()
    neuron_rest = IzhikevichNeuron()
    net_rest = Network(neuron_rest.group, neuron_rest.state_monitor)
    net_rest.run(100*ms)
    
    # --- Test 2: The Rheobase Fork (Sub vs Super Threshold) ---
    start_scope()
    # Create two neurons in the same group to compare them
    neurons_rheo = IzhikevichNeuron(N=2) 
    net_rheo = Network(neurons_rheo.group, neurons_rheo.state_monitor)
    
    net_rheo.run(20*ms) # Rest
    neurons_rheo.group.I[0] = 3.9 # Sub-threshold (Should not fire)
    neurons_rheo.group.I[1] = 4.1 # Super-threshold (Should fire)
    net_rheo.run(480*ms)
    
    # --- Test 3: Active Adaptation (Burst to Steady State) ---
    start_scope()
    neuron_active = IzhikevichNeuron()
    net_active = Network(neuron_active.group, neuron_active.state_monitor, neuron_active.spike_monitor)
    
    net_active.run(20*ms)
    neuron_active.inject_current(50.0) # Higher current for clearer burst-to-steady transition
    net_active.run(480*ms)

    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Rest
    ax1.plot(neuron_rest.state_monitor.t/ms, neuron_rest.state_monitor.v[0], color='blue', linewidth=2)
    ax1.axhline(-70, color='red', linestyle='--', alpha=0.6, label='Equilibrium (-70mV)')
    ax1.set_title('Test 1: Baseline Resting Potential (I=0)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_ylim(-75, -55)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Plot 2: Rheobase Comparison
    ax2.plot(neurons_rheo.state_monitor.t/ms, neurons_rheo.state_monitor.v[0], color='red', label='Sub-Threshold (I=3.9)')
    ax2.plot(neurons_rheo.state_monitor.t/ms, neurons_rheo.state_monitor.v[1], color='green', label='Super-Threshold (I=4.1)')
    ax2.set_title('Test 2: Rheobase Effect (The Threshold Fork)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.axvline(20, color='black', linestyle=':', label='Current Injection Start')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Plot 3: Adaptation
    ax3.plot(neuron_active.state_monitor.t/ms, neuron_active.state_monitor.v[0], color='purple')
    ax3.set_title('Test 3: Spike Frequency Adaptation (I=15.0)')
    ax3.set_ylabel('Voltage (mV)')
    ax3.set_xlabel('Time (ms)')
    ax3.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("01_CA1_Baseline_Comparison.png", dpi=300)
    
    # Exporting spikes for Test 3 to verify the inter-spike interval (ISI) slowing
    df_spikes = pd.DataFrame({'Spike_Times_ms': neuron_active.spike_monitor.t/ms})
    df_spikes.to_csv("Phase1_ISI_Verification.csv", index=False)
    
    print("Saved -> '01_CA1_Baseline_Comparison.png' and 'Phase1_ISI_Verification.csv'")

def run_forced_stdp_control():
    print("\n--- Running Phase 2: Forced STDP (LTP & LTD) ---")
    start_scope()
    
    neurons = IzhikevichNeuron(N=4)
    
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    '''
    on_pre_eqs = '''
    v_post += w                
    apre += 2.0              
    w = clip(w + apost, 0.0, 10.0) 
    '''
    on_post_eqs = '''
    apost += -2.0            
    w = clip(w + apre, 0.0, 10.0)  
    '''
    
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
    syn.connect(i=0, j=1) 
    syn.connect(i=2, j=3) 
    syn.w = 5.0 
    
    syn_monitor = StateMonitor(syn, 'w', record=True)
    net = Network(neurons.group, neurons.state_monitor, neurons.spike_monitor, syn, syn_monitor)

    net.run(20*ms)
    
    # LTP Pair
    neurons.group.I[0] = 50.0  
    net.run(2*ms)
    neurons.group.I[0] = 0.0 
    net.run(8*ms) 
    neurons.group.I[1] = 50.0  
    net.run(2*ms)
    neurons.group.I[1] = 0.0 
    net.run(30*ms) 
    
    # LTD Pair
    neurons.group.I[3] = 50.0  
    net.run(2*ms)
    neurons.group.I[3] = 0.0 
    net.run(8*ms) 
    neurons.group.I[2] = 50.0  
    net.run(2*ms)
    neurons.group.I[2] = 0.0 
    net.run(30*ms)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(syn_monitor.t/ms, syn_monitor.w[0], color='blue', linewidth=3, label='LTP (Pair 1: Pre before Post)')
    ax.plot(syn_monitor.t/ms, syn_monitor.w[1], color='red', linewidth=3, label='LTD (Pair 2: Post before Pre)')
    ax.set_title('Forced STDP: Synaptic Weight Changes')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Synaptic Weight ($w$)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("02_CA1_Forced_STDP.png", dpi=300)
    plt.close('all')
    print("Saved -> '02_CA1_Forced_STDP.png'")

def run_emergent_stdp():
    print("\n--- Running Phase 3: Emergent STDP (Spatial Summation) ---")
    start_scope()
    
    neurons = IzhikevichNeuron(N=5) 
    
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    '''
    on_pre_eqs = '''
    v_post += w                
    apre += 2.025              
    w = clip(w + apost, 0.0, 10.0) 
    '''
    on_post_eqs = '''
    apost += -2.0            
    w = clip(w + apre, 0.0, 10.0)  
    '''
    
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
    
    for i in range(4):
        syn.connect(i=i, j=4) 
    syn.w = 4.5 
    
    syn_monitor = StateMonitor(syn, 'w', record=True) 
    net = Network(neurons.group, neurons.state_monitor, neurons.spike_monitor, syn, syn_monitor)

    net.run(20*ms)
    
    for i in range(4):
        neurons.group.I[i] = 50.0  
    net.run(2*ms)
    for i in range(4):
        neurons.group.I[i] = 0.0 
    net.run(40*ms) 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[0], color='black', label='Pre-Synaptic (Summating)')
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[4], color='blue', label='Post-Synaptic (Natural Spike)')
    ax1.set_title('Emergent Spiking via Spatial Summation')
    ax1.set_ylabel('Membrane Voltage (mV)')
    ax1.set_xlabel('Time (ms)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    ax2.plot(syn_monitor.t/ms, syn_monitor.w[0], color='green', linewidth=3, label='Synaptic Weight ($w$)')
    ax2.set_title('Synaptic Weight ($w$ jumps to exactly 145% of baseline)')
    ax2.set_ylabel('Synaptic Weight ($w$)')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("03_CA1_Emergent_STDP.png", dpi=300)
    plt.close('all')
    print("Saved -> '03_CA1_Emergent_STDP.png'")

def run_es_coupling_failure_proof():
    print("\n--- Running Phase 4: E-S Coupling Failure Proof ---")
    start_scope()
    
    neurons = IzhikevichNeuron(N=2)
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    '''
    on_pre_eqs = '''
    v_post += w                
    apre += 2.0              
    w = clip(w + apost, 0.0, 10.0) 
    '''
    on_post_eqs = '''
    apost += -2.0            
    w = clip(w + apre, 0.0, 10.0)  
    '''
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
    syn.connect(i=0, j=1)
    syn.w = 4.0 
    
    net = Network(neurons.group, neurons.state_monitor, syn)

    # Pre-Learning
    net.run(20*ms)
    neurons.group.I[1] = 3.9  
    net.run(100*ms)
    neurons.group.I[1] = 0.0 
    net.run(20*ms)

    # Trigger LTP
    neurons.group.I[0] = 50.0  
    net.run(2*ms)
    neurons.group.I[0] = 0.0 
    net.run(8*ms) 
    neurons.group.I[1] = 50.0  
    net.run(2*ms)
    neurons.group.I[1] = 0.0 
    net.run(50*ms) 

    # Post-Learning
    neurons.group.I[1] = 3.9  
    net.run(100*ms)
    neurons.group.I[1] = 0.0 
    net.run(20*ms)

    # Dynamic Time Masking (immune to indexing errors)
    t_ms = neurons.state_monitor.t / ms
    v_post = neurons.state_monitor.v[1]
    
    mask_pre = (t_ms >= 20) & (t_ms <= 120)
    mask_post = (t_ms >= 202) & (t_ms <= 302)
    
    pre_v = v_post[mask_pre]
    post_v = v_post[mask_post]
    post_v = post_v[:len(pre_v)] # Ensure exact length match
    
    time_slice = np.linspace(0, 100, len(pre_v))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_slice, pre_v, label='Pre-Learning Response (I=3.9)', color='blue', linewidth=4, alpha=0.5)
    ax.plot(time_slice, post_v, label='Post-Learning Response (I=3.9)', color='red', linestyle='--')
    ax.set_title('Standard Model Limitation: Static Intrinsic Excitability')
    ax.set_xlabel('Time during stimulus injection (ms)')
    ax.set_ylabel('Membrane Voltage (mV)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("04_CA1_Standard_Failure.png", dpi=300)
    plt.close('all')
    print("Saved -> '04_CA1_Standard_Failure.png'")

if __name__ == '__main__':
    print("Starting CA1 Experiment Suite...")
    
    try:
        run_baseline_validation()
    except Exception as e:
        print(f"FAILED Phase 1: {e}")
        
    try:
        run_forced_stdp_control()
    except Exception as e:
        print(f"FAILED Phase 2: {e}")
        
    try:
        run_emergent_stdp()
    except Exception as e:
        print(f"FAILED Phase 3: {e}")
        
    try:
        run_es_coupling_failure_proof()
    except Exception as e:
        print(f"FAILED Phase 4: {e}")