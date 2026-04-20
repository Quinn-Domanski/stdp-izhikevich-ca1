import matplotlib.pyplot as plt
from brian2 import *
from model import IzhikevichNeuron

def run_standard_stdp():
    print("Running Standard STDP Control Experiment...")
    start_scope()
    
    # 1. Initialize our two biologically-tuned CA1 neurons
    neurons = IzhikevichNeuron(N=2, b=0.2347, c=-66.5)

    # 2. STDP Learning Parameters
    tau_pre = 20*ms    # 20ms memory window for the Pre-spike
    tau_post = 20*ms   # 20ms memory window for the Post-spike
    A_pre = 1.5        # The "Jump" in connection strength if LTP happens
    A_post = -1.5      # The "Drop" in connection strength if LTD happens
    w_max = 10.0       # The absolute maximum synaptic weight
    
    # 3. The Synapse Math
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    
    on_pre_eqs = '''
    v_post += w                # Deliver the EPSP
    apre += A_pre              # Start the Pre-timer
    w = clip(w + apost, 0, w_max) # Check for LTD (Unlearning)
    '''
    
    on_post_eqs = '''
    apost += A_post            # Start the Post-timer
    w = clip(w + apre, 0, w_max)  # Check for LTP (Learning)
    '''
    
    # 4. Build and Connect
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, 
                   on_pre=on_pre_eqs, on_post=on_post_eqs)
    syn.connect(i=0, j=1)
    syn.w = 4.0 # Baseline starting weight
    
    # We need a special monitor just to watch 'w' change!
    syn_monitor = StateMonitor(syn, 'w', record=True)
    
    net = Network(neurons.group, neurons.state_monitor, neurons.spike_monitor, syn, syn_monitor)

    # ---------------------------------------------------------
    # THE EXPERIMENT TIMELINE
    # ---------------------------------------------------------
    net.run(20*ms) # Rest
    
    print("Firing Pre-synaptic neuron...")
    neurons.group.I[0] = 50.0 
    net.run(2*ms)
    neurons.group.I[0] = 0.0 
    
    print("Waiting 10ms...")
    net.run(10*ms) 
    
    print("Firing Post-synaptic neuron to trigger STDP!")
    neurons.group.I[1] = 50.0 
    net.run(2*ms)
    neurons.group.I[1] = 0.0 
    
    net.run(30*ms) # Let it settle

    # ---------------------------------------------------------
    # Plotting the Learning
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top Graph: The Spikes
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[0], label='Pre-Synaptic', color='black')
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[1], label='Post-Synaptic', color='#1f77b4')
    ax1.set_title('Neural Activity: Pre fires, then Post fires 10ms later')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend()
    
    # Bottom Graph: The Synaptic Weight (Learning)
    ax2.plot(syn_monitor.t/ms, syn_monitor.w[0], color='red', linewidth=3)
    ax2.set_title('Synaptic Weight ($w$)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Connection Strength')
    
    plt.tight_layout()
    plt.savefig("CA1_Standard_STDP.png", dpi=300)
    print("Saved as 'CA1_Standard_STDP.png'!")

if __name__ == '__main__':
    run_standard_stdp()