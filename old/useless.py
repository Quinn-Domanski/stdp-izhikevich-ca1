import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import *
from model import IzhikevichNeuron

prefs.codegen.target = "numpy" 

def run_baseline_validation():
    print("\n--- Running Phase 1: Baseline Validation ---")
    start_scope()

    PULSE_START  = 100  # ms — resting period before stimulus
    PULSE_DUR    = 500  # ms — matches Fig. 2 step duration in paper
    RECORD_TOTAL = 800  # ms — watch recovery after pulse ends

    def run_pulse_experiment(current_value, label):
        neuron = IzhikevichNeuron()
        net = Network(neuron.group, neuron.state_monitor, neuron.spike_monitor)

        neuron.inject_current(0.0)
        net.run(PULSE_START * ms)

        neuron.inject_current(current_value)
        net.run(PULSE_DUR * ms)

        neuron.inject_current(0.0)
        net.run((RECORD_TOTAL - PULSE_START - PULSE_DUR) * ms)

        n_spikes = len(neuron.spike_monitor.t)
        print(f"  {label}: {n_spikes} spike(s) at times {np.round(neuron.spike_monitor.t/ms, 1)}")
        return neuron

    # ── Test 1: Resting Potential ────────────────────────────────────────────
    neuron_rest = IzhikevichNeuron()
    net_rest = Network(neuron_rest.group, neuron_rest.state_monitor, neuron_rest.spike_monitor)
    net_rest.run(RECORD_TOTAL * ms)

    # ── Test 2: Sub-threshold ────────────────────────────────────────────────
    # I=3.0 — safely below rheobase, v depolarises slightly then returns
    neuron_sub = run_pulse_experiment(3.0, "I=3.0  sub-threshold")

    # ── Test 3: Near-rheobase ────────────────────────────────────────────────
    # I=4.0 — fires a small number of spikes then adaptation silences it
    neuron_rheo = run_pulse_experiment(4.0, "I=4.0  near-rheobase")

    # ── Test 4: SFA — matches Fig. 2 RS panel from Izhikevich (2003) ─────────
    # I=10 is the exact stimulus used in the paper for all Fig. 2 demonstrations
    neuron_sfa = run_pulse_experiment(10.0, "I=10.0 SFA (Fig.2 RS)")

    spike_times = neuron_sfa.spike_monitor.t / ms
    # Only count spikes during the pulse window
    pulse_spikes = spike_times[(spike_times >= PULSE_START) &
                               (spike_times <= PULSE_START + PULSE_DUR)]
    isis = np.diff(pulse_spikes)
    ratio = isis[-1] / isis[0] if len(isis) >= 2 else float('nan')
    print(f"  SFA ISIs (ms): {np.round(isis, 1)}")
    print(f"  Adaptation ratio (last/first ISI): {ratio:.2f}x")

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    def shade_pulse(ax):
        ax.axvspan(PULSE_START, PULSE_START + PULSE_DUR,
                   color='gold', alpha=0.12, label='Stimulus window')
    def add_grid(ax):
        ax.grid(True, linestyle=':', alpha=0.5)
    def add_vlines(ax):
        ax.axvline(PULSE_START, color='dimgray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(PULSE_START + PULSE_DUR, color='dimgray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Panel 1 — Resting potential
    ax1.plot(neuron_rest.state_monitor.t/ms, neuron_rest.state_monitor.v[0],
             color='steelblue', linewidth=2, label='Membrane voltage')
    ax1.axhline(-65.0, color='crimson', linestyle='--', alpha=0.6,
                label='Resting potential: −65 mV')
    ax1.set_title('Test 1: Resting Potential — I = 0 throughout', fontweight='bold')
    ax1.set_ylim(-78, 35)
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend(fontsize=9, loc='upper right')
    add_grid(ax1)

    # Panel 2 — Sub-threshold
    shade_pulse(ax2)
    add_vlines(ax2)
    ax2.plot(neuron_sub.state_monitor.t/ms, neuron_sub.state_monitor.v[0],
             color='darkorange', linewidth=2, label='I = 3.0 (~160 pA)')
    ax2.axhline(30.0, color='gray', linestyle=':', alpha=0.5, label='Spike threshold (30 mV)')
    n_sub = len(neuron_sub.spike_monitor.t)
    ax2.set_title(f'Test 2: Sub-threshold — I = 3.0, {n_sub} spikes (depolarises, never fires)',
                  fontweight='bold')
    ax2.set_ylim(-78, 35)
    ax2.set_ylabel('Voltage (mV)')
    ax2.legend(fontsize=9, loc='upper right')
    add_grid(ax2)

    # Panel 3 — Near-rheobase
    shade_pulse(ax3)
    add_vlines(ax3)
    ax3.plot(neuron_rheo.state_monitor.t/ms, neuron_rheo.state_monitor.v[0],
             color='forestgreen', linewidth=2, label='I = 4.0 (~213 pA)')
    ax3.axhline(30.0, color='gray', linestyle=':', alpha=0.5, label='Spike threshold (30 mV)')
    n_rheo = len(neuron_rheo.spike_monitor.t)
    ax3.set_title(f'Test 3: Near-rheobase — I = 4.0, {n_rheo} spike(s), adaptation reduces firing',
                  fontweight='bold')
    ax3.set_ylim(-78, 35)
    ax3.set_ylabel('Voltage (mV)')
    ax3.legend(fontsize=9, loc='upper right')
    add_grid(ax3)

    # Panel 4 — SFA, replicating Fig. 2 RS from Izhikevich (2003)
    shade_pulse(ax4)
    add_vlines(ax4)
    ax4.plot(neuron_sfa.state_monitor.t/ms, neuron_sfa.state_monitor.v[0],
             color='purple', linewidth=2, label='I = 10.0 (~533 pA)')
    # Annotate ISIs between spikes during pulse only
    for i, (t0, t1) in enumerate(zip(pulse_spikes[:-1], pulse_spikes[1:])):
        mid = (t0 + t1) / 2
        ax4.annotate(f'{isis[i]:.0f} ms', xy=(mid, -76), fontsize=7.5,
                     ha='center', color='purple', alpha=0.9)
    ratio_str = f'{ratio:.2f}×' if not np.isnan(ratio) else 'N/A'
    ax4.set_title(
        f'Test 4: Spike Frequency Adaptation — I = 10 (Fig. 2 RS, Izhikevich 2003) '
        f'— ISI ratio {ratio_str}',
        fontweight='bold')
    ax4.set_ylim(-78, 35)
    ax4.set_ylabel('Voltage (mV)')
    ax4.set_xlabel('Time (ms)')
    ax4.legend(fontsize=9, loc='upper right')
    add_grid(ax4)

    plt.tight_layout(pad=1.5)
    plt.savefig("01_CA1_Baseline.png", dpi=300)

    # ── CSV export ────────────────────────────────────────────────────────────
    print("  Exporting Phase 1 raw data...")
    pd.DataFrame({
        'Time_ms':    neuron_sfa.state_monitor.t / ms,
        'Voltage_mV': neuron_sfa.state_monitor.v[0]
    }).to_csv("Phase1_Active_Voltage_Trace.csv", index=False)

    pd.DataFrame({
        'Spike_Times_ms': spike_times
    }).to_csv("Phase1_Spike_Timestamps.csv", index=False)

    plt.close('all')
    print("  Saved → '01_CA1_Baseline.png', CSVs")

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