import matplotlib.pyplot as plt
from brian2 import *
from model import ContinuousIPNeuron, DiscreteIPNeuron

prefs.codegen.target = "numpy" 

def plot_results(state_mon, syn_mon, title, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(state_mon.t/ms, state_mon.v[0], color='gray', linestyle=':', linewidth=1.5, label='Pre')
    ax1.plot(state_mon.t/ms, state_mon.v[1], color='blue', linewidth=2, label='Post')
    
    # Highlight the 4 specific testing/training zones
    ax1.axvspan(20, 170, color='red', alpha=0.1, label='Stimulus: I=3.7 (Fails)')
    ax1.axvspan(220, 232, color='purple', alpha=0.2, label='Training')
    ax1.axvspan(312, 462, color='green', alpha=0.1, label='Stimulus: I=3.7 (Succeeds)')
    ax1.axvspan(662, 812, color='orange', alpha=0.1, label='Memory Test: I=3.7')
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
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

    ax3 = ax2.twinx()  
    color = 'tab:orange'
    ax3.set_ylabel('Excitability ($E$)', color=color)  
    ax3.plot(state_mon.t/ms, state_mon.E[1], color=color, linewidth=2, linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close('all')
    print(f"Saved -> '{filename}'")

def run_continuous():
    start_scope()
    neurons = ContinuousIPNeuron(N=2)
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    w_total_post = w : 1 (summed)  
    '''
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, 
                   on_pre='v_post += w; apre += 2.0; w = clip(w + apost, 0.0, 10.0)', 
                   on_post='apost += -2.0; w = clip(w + apre, 0.0, 10.0)')
    syn.connect(i=0, j=1); syn.w = 5.0 
    syn_mon = StateMonitor(syn, 'w', record=True)
    net = Network(neurons.group, neurons.state_monitor, syn, syn_mon)
    
    # We now pass the syn object so the timeline can manipulate it
    run_timeline(net, neurons.group, syn)
    plot_results(neurons.state_monitor, syn_mon, 'Continuous IP (Faucet Method)', '04_Continuous_IP.png')

def run_discrete():
    start_scope()
    neurons = DiscreteIPNeuron(N=2)
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    '''
    on_pre_eqs = '''
    v_post += w                
    apre += 2.0              
    w = clip(w + apost, 0.0, 10.0)
    E_post = clip(E_post + (apost * 0.05), 0.0, 0.5) 
    '''
    on_post_eqs = '''
    apost += -2.0            
    w = clip(w + apre, 0.0, 10.0)
    E_post = clip(E_post + (apre * 0.05), 0.0, 0.5)
    '''
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, on_pre=on_pre_eqs, on_post=on_post_eqs)
    syn.connect(i=0, j=1); syn.w = 5.0 
    syn_mon = StateMonitor(syn, 'w', record=True)
    net = Network(neurons.group, neurons.state_monitor, syn, syn_mon)
    
    run_timeline(net, neurons.group, syn)
    plot_results(neurons.state_monitor, syn_mon, 'Discrete IP (Event-Driven Method)', '04_Discrete_IP.png')

def run_timeline(net, group, syn):
    # 1. "Before" Test
    net.run(20*ms); group.I[1] = 3.7
    net.run(150*ms); group.I[1] = 0.0
    net.run(50*ms) 
    
    # 2. Training (LTP)
    group.I[0] = 50.0; net.run(2*ms); group.I[0] = 0.0; net.run(8*ms) 
    group.I[1] = 50.0; net.run(2*ms); group.I[1] = 0.0; net.run(80*ms) 
    
    # 3. "After" Test (Immediate)
    group.I[1] = 3.7; net.run(150*ms); group.I[1] = 0.0
    
    # --- NEW PHASE ---
    # 4. Long Rest & Final Test
    syn.w = 5.0  # Reset synapse to force the continuous model to drain
    net.run(200*ms) 
    
    group.I[1] = 3.7; net.run(150*ms); group.I[1] = 0.0
    net.run(300*ms)

if __name__ == '__main__':
    run_continuous()
    run_discrete()