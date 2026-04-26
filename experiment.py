import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import *
from model import IzhikevichNeuron
from model import FluidIPNeuron

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
    neurons_rheo.group.I[0] = 3.7 # Sub-threshold (Should not fire)
    neurons_rheo.group.I[1] = 4.0 # Super-threshold (Should fire)
    net_rheo.run(480*ms)
    
    # --- Test 3: Active Adaptation (Burst to Steady State) ---
    start_scope()
    neuron_active = IzhikevichNeuron()
    net_active = Network(neuron_active.group, neuron_active.state_monitor, neuron_active.spike_monitor)
    
    neuron_active.inject_current(50.0) # Higher current for clearer burst-to-steady transition
    net_active.run(50*ms)

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
    ax2.plot(neurons_rheo.state_monitor.t/ms, neurons_rheo.state_monitor.v[0], color='red', label='Sub-Threshold (I=3.7)')
    ax2.plot(neurons_rheo.state_monitor.t/ms, neurons_rheo.state_monitor.v[1], color='green', label='Super-Threshold (I=4.0)')
    ax2.set_title('Test 2: Rheobase Effect (The Threshold Fork)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.axvline(20, color='black', linestyle=':', label='Current Injection Start')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Plot 3: Adaptation
    ax3.plot(neuron_active.state_monitor.t/ms, neuron_active.state_monitor.v[0], color='purple')
    ax3.set_title('Test 3: Spike Frequency Adaptation (I=50.0)')
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

def run_phase3_control():
    print("\n--- Running Phase 3: Standard STDP Control (No Intrinsic Plasticity) ---")
    start_scope()

    # Import the standard, static Phase 1 cell body
    neurons = IzhikevichNeuron(N=2)

    # Standard STDP without the bridge. The cell body cannot 'see' w.
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
    syn.w = 5.0 
    
    syn_mon = StateMonitor(syn, 'w', record=True)
    net = Network(neurons.group, neurons.state_monitor, syn, syn_mon)

    # --- THE EXPERIMENTAL TIMELINE ---
    
    # Event 1: The "Before" Test (Current = 3.7)
    net.run(20*ms)
    neurons.group.I[1] = 3.7
    net.run(150*ms)
    neurons.group.I[1] = 0.0
    net.run(50*ms) 

    # Event 2: STDP Training (Force Pre, then Post to trigger LTP)
    neurons.group.I[0] = 50.0  
    net.run(2*ms)
    neurons.group.I[0] = 0.0 
    net.run(8*ms) 
    neurons.group.I[1] = 50.0  
    net.run(2*ms)
    neurons.group.I[1] = 0.0 
    net.run(80*ms) 

    # Event 3: The "After" Test (Current = 3.7)
    neurons.group.I[1] = 3.7
    net.run(150*ms)
    neurons.group.I[1] = 0.0
    net.run(50*ms)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[0], color='gray', linestyle=':', linewidth=1.5, label='Pre-Synaptic (Forced)')
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[1], color='blue', linewidth=2, label='Post-Synaptic Voltage')
    
    ax1.axvspan(20, 170, color='red', alpha=0.1, label='Stimulus: I=3.7 (Fails)')
    ax1.axvspan(220, 232, color='purple', alpha=0.2, label='STDP Training')
    ax1.axvspan(312, 462, color='red', alpha=0.1, label='Stimulus: I=3.7 (Fails Again)')
    
    ax1.set_title('Phase 3 Control: Standard STDP Fails to Alter Intrinsic Excitability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_ylim(-75, 40)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    color = 'tab:green'
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Synaptic Weight ($w$)', color=color)
    ax2.plot(syn_mon.t/ms, syn_mon.w[0], color=color, linewidth=3, label='Weight Increases (LTP)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(4.5, 7.5)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("03_CA1_Control_Failure.png", dpi=300)
    plt.close('all')
    print("Saved -> '03_CA1_Control_Failure.png'")

def run_phase4_intrinsic_plasticity():
    print("\n--- Running Phase 4: Emergent Intrinsic Plasticity (Input Resistance Model) ---")
    start_scope()

    # 1. IP-Enabled Neuron Equations using R_gain
    ip_eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I * R_gain) / ms : 1
    du/dt = (a*(b*v - u)) / ms : 1
    
    # R_gain acts as a multiplier on the incoming current, mapping to Input Resistance.
    R_gain = 1.0 + k * delta_w : 1 
    
    # We clip the delta so that 0-input pre-neurons aren't mathematically punished.
    delta_w = clip(w_total - w_base, 0.0, 10.0) : 1
    
    w_total : 1
    I : 1
    a : 1
    b : 1
    c : 1
    d : 1
    k : 1
    w_base : 1
    '''
    
    neurons = NeuronGroup(2, model=ip_eqs, threshold='v >= 30', reset='v = c; u = u + d', method='euler')
    
    # Parameters strictly matching your Phase 1 setup
    neurons.a = 0.02
    neurons.b = 0.2
    neurons.c = -65.0
    neurons.d = 8.0
    neurons.w_base = 5.0
    neurons.k = 0.05  
    
    # THE FIX: Restoring your exact Phase 1 initial conditions
    # This prevents the initial momentum overshoot and false spike
    neurons.v = -65.0
    neurons.u = -13.0
    neurons.I = 0.0
    neurons.w_total = 5.0 
    
    # 2. IP-Enabled Synapse Equations
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    w_total_post = w : 1 (summed)  
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
    
    syn = Synapses(neurons, neurons, model=syn_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
    syn.connect(i=0, j=1) 
    syn.w = 5.0 
    
    state_mon = StateMonitor(neurons, ['v', 'R_gain'], record=True)
    syn_mon = StateMonitor(syn, 'w', record=True)
    net = Network(neurons, state_mon, syn, syn_mon)

    # --- THE EXPERIMENTAL TIMELINE ---
    
    net.run(20*ms)
    neurons.I[1] = 3.7
    net.run(150*ms)
    neurons.I[1] = 0.0
    net.run(50*ms) 

    neurons.I[0] = 50.0  
    net.run(2*ms)
    neurons.I[0] = 0.0 
    net.run(8*ms) 
    neurons.I[1] = 50.0  
    net.run(2*ms)
    neurons.I[1] = 0.0 
    net.run(80*ms) 

    neurons.I[1] = 3.7
    net.run(150*ms)
    neurons.I[1] = 0.0
    net.run(50*ms)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(state_mon.t/ms, state_mon.v[0], color='gray', linestyle=':', linewidth=1.5, label='Pre-Synaptic (Forced)')
    ax1.plot(state_mon.t/ms, state_mon.v[1], color='blue', linewidth=2, label='Post-Synaptic Voltage')
    
    ax1.axvspan(20, 170, color='red', alpha=0.1, label='Stimulus: I=3.7 (Sub-Threshold)')
    ax1.axvspan(220, 232, color='purple', alpha=0.2, label='Forced STDP Training (LTP)')
    ax1.axvspan(312, 462, color='green', alpha=0.1, label='Stimulus: I=3.7 (Now Super-Threshold)')
    
    ax1.set_title('Phase 4: E-S Coupling Enhancement via Input Resistance ($R_{gain}$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_ylim(-75, 40)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    color = 'tab:green'
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Synaptic Weight ($w$)', color=color)
    ax2.plot(syn_mon.t/ms, syn_mon.w[0], color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Plot the R_gain multiplier 
    ax3 = ax2.twinx()  
    color = 'tab:orange'
    ax3.set_ylabel('Input Gain Multiplier ($R_{gain}$)', color=color)  
    ax3.plot(state_mon.t/ms, state_mon.R_gain[1], color=color, linewidth=2, linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig("04_CA1_IP_Hero_Graph.png", dpi=300)
    plt.close('all')
    print("Saved -> '04_CA1_IP_Hero_Graph.png'")

def run_phase4_perfect_ip():
    print("\n--- Running Phase 4: Perfect E-S Coupling (Network Bridge) ---")
    start_scope()

    # 1. IMPORT YOUR EXACT PHASE 1 MODEL
    # We do not overwrite your physics. Your baseline is perfectly protected.
    neurons = IzhikevichNeuron(N=2)
    
    # 2. STANDARD STDP SYNAPSE
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
    syn.w = 5.0 

    # 3. THE INTRINSIC PLASTICITY BRIDGE
    # We track the external stimulus you intend to inject
    base_I = np.zeros(2)
    gain_history = [] # To graph the underlying math
    
    # This operation runs every millisecond, acting as the biological E-S bridge
    @network_operation(dt=defaultclock.dt)
    def apply_es_coupling():
        current_w = syn.w[0]
        delta_w = max(current_w - 5.0, 0.0)
        
        # Calculate the Input Resistance Amplifier (scales up as the synapse learns)
        R_gain = 1.0 + 0.04 * delta_w
        
        # Inject the amplified current. 
        # If base_I is 0, the cell feels 0. Resting potential is perfectly protected.
        neurons.group.I[0] = base_I[0]
        neurons.group.I[1] = base_I[1] * R_gain
        
        gain_history.append(R_gain)

    # Monitors
    syn_mon = StateMonitor(syn, 'w', record=True)
    net = Network(neurons.group, neurons.state_monitor, syn, syn_mon, apply_es_coupling)

    # --- THE EXPERIMENTAL TIMELINE ---
    
    # Event 1: The "Before" Test (Will use base Phase 1 physics)
    net.run(20*ms)
    base_I[1] = 3.7
    net.run(150*ms)
    base_I[1] = 0.0
    net.run(50*ms) 

    # Event 2: Forced STDP Training
    base_I[0] = 50.0  
    net.run(2*ms)
    base_I[0] = 0.0 
    net.run(8*ms) 
    base_I[1] = 50.0  
    net.run(2*ms)
    base_I[1] = 0.0 
    net.run(80*ms) 

    # Event 3: The "After" Test (Will use Amplified physics)
    base_I[1] = 3.7
    net.run(150*ms)
    base_I[1] = 0.0
    net.run(50*ms)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[0], color='gray', linestyle=':', linewidth=1.5, label='Pre-Synaptic')
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[1], color='blue', linewidth=2, label='Post-Synaptic')
    
    ax1.axvspan(20, 170, color='red', alpha=0.1, label='Stimulus: I=3.7 (Fails)')
    ax1.axvspan(220, 232, color='purple', alpha=0.2, label='Training (LTP)')
    ax1.axvspan(312, 462, color='green', alpha=0.1, label='Stimulus: I=3.7 (Succeeds)')
    
    ax1.set_title('Phase 4: Perfect E-S Coupling via Dynamic Input Resistance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_ylim(-75, 40)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    color = 'tab:green'
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Synaptic Weight ($w$)', color=color)
    ax2.plot(syn_mon.t/ms, syn_mon.w[0], color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Plot the Amplifier (R_gain)
    ax3 = ax2.twinx()  
    color = 'tab:orange'
    ax3.set_ylabel('Input Gain Multiplier ($R_{gain}$)', color=color)  
    
    # We trim gain_history to match the exact length of the time array to prevent plotting errors
    time_array = neurons.state_monitor.t/ms
    ax3.plot(time_array, gain_history[:len(time_array)], color=color, linewidth=2, linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig("04_CA1_Perfect_IP_Graph.png", dpi=300)
    plt.close('all')
    print("Saved -> '04_CA1_Perfect_IP_Graph.png'")

# def run_phase4_fluid_ip():
#     print("\n--- Running Phase 4: Fluid E-S Coupling (Generalized Model) ---")
#     start_scope()

#     # 1. Import your custom Fluid IP Neuron
#     neurons = FluidIPNeuron(N=2)
    
#     # 2. IP-Enabled Synapse Equations (Dimensionless)
#     syn_eqs = '''
#     w : 1
#     dapre/dt = -apre/(20*ms) : 1 (event-driven)
#     dapost/dt = -apost/(20*ms) : 1 (event-driven)
#     w_total_post = w : 1 (summed)  
#     '''
#     on_pre_eqs = '''
#     v_post += w               
#     apre += 2.0              
#     w = clip(w + apost, 0.0, 10.0) 
#     '''
#     on_post_eqs = '''
#     apost += -2.0            
#     w = clip(w + apre, 0.0, 10.0)  
#     '''
    
#     syn = Synapses(neurons.group, neurons.group, model=syn_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
#     syn.connect(i=0, j=1) 
#     syn.w = 5.0 
    
#     state_mon = StateMonitor(neurons.group, ['v', 'v_t'], record=True)
#     syn_mon = StateMonitor(syn, 'w', record=True)
#     net = Network(neurons.group, state_mon, syn, syn_mon)

#     # --- THE EXPERIMENTAL TIMELINE ---
    
#     # Event 1: The "Before" Test (Inject 140. Fails because Rheobase is ~157)
#     net.run(20*ms)
#     neurons.group.I[1] = 140.0
#     net.run(150*ms)
#     neurons.group.I[1] = 0.0
#     net.run(50*ms) 

#     # Event 2: Forced STDP Training
#     neurons.group.I[0] = 500.0  
#     net.run(2*ms)
#     neurons.group.I[0] = 0.0 
#     net.run(8*ms) 
#     neurons.group.I[1] = 500.0  
#     net.run(2*ms)
#     neurons.group.I[1] = 0.0 
#     net.run(80*ms) 

#     # Event 3: The "After" Test (Inject 140. Succeeds because Rheobase dropped)
#     neurons.group.I[1] = 140.0
#     net.run(150*ms)
#     neurons.group.I[1] = 0.0
#     net.run(50*ms)

#     # --- PLOTTING ---
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
#     ax1.plot(state_mon.t/ms, state_mon.v[0], color='gray', linestyle=':', linewidth=1.5, label='Pre-Synaptic (Forced)')
#     ax1.plot(state_mon.t/ms, state_mon.v[1], color='blue', linewidth=2, label='Post-Synaptic Voltage')
    
#     ax1.axvspan(20, 170, color='red', alpha=0.1, label='Stimulus: I=140 (Fails)')
#     ax1.axvspan(220, 232, color='purple', alpha=0.2, label='Forced STDP Training (LTP)')
#     ax1.axvspan(312, 462, color='green', alpha=0.1, label='Stimulus: I=140 (Succeeds)')
    
#     ax1.set_title('Phase 4: Fluid E-S Coupling (Generalized Model)', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('Voltage (mV)')
#     ax1.set_ylim(-75, 40)
#     ax1.legend(loc='upper left')
#     ax1.grid(True, linestyle=':', alpha=0.6)

#     color = 'tab:green'
#     ax2.set_xlabel('Time (ms)')
#     ax2.set_ylabel('Synaptic Weight ($w$)', color=color)
#     ax2.plot(syn_mon.t/ms, syn_mon.w[0], color=color, linewidth=3)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.grid(True, linestyle=':', alpha=0.6)

#     ax3 = ax2.twinx()  
#     color = 'tab:orange'
#     ax3.set_ylabel('Dynamic Threshold ($v_t$)', color=color)  
#     ax3.plot(state_mon.t/ms, state_mon.v_t[1], color=color, linewidth=2, linestyle='--')
#     ax3.tick_params(axis='y', labelcolor=color)

#     plt.tight_layout()
#     plt.savefig("04_CA1_Fluid_IP_Graph.png", dpi=300)
#     plt.close('all')
#     print("Saved -> '04_CA1_Fluid_IP_Graph.png'")

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
    

    run_phase3_control()
    run_phase4_intrinsic_plasticity()
    run_phase4_perfect_ip()
    run_phase4_fluid_ip()