import matplotlib.pyplot as plt
from brian2 import *
from model import IzhikevichNeuron

def run_stdp_learning_suite():
    print("Running Full STDP Suite (LTP and LTD)...")
    start_scope()
    
    # 1. Initialize 4 biologically-tuned CA1 neurons
    # Pair 1 (LTP): Neuron 0 (Pre), Neuron 1 (Post)
    # Pair 2 (LTD): Neuron 2 (Pre), Neuron 3 (Post)
    neurons = IzhikevichNeuron(N=4, b=0.2347, c=-66.5)

    # 2. STDP Learning Parameters
    tau_pre = 20*ms    
    tau_post = 20*ms   
    A_pre = 2.0        # Jump UP in connection strength (LTP)
    A_post = -2.0      # Drop DOWN in connection strength (LTD)
    w_max = 10.0       
    w_min = 0.0        
    
    # 3. The Synapse Math (Traces)
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    
    on_pre_eqs = '''
    v_post += w                
    apre += A_pre              
    w = clip(w + apost, w_min, w_max) # LTD Check
    '''
    
    on_post_eqs = '''
    apost += A_post            
    w = clip(w + apre, w_min, w_max)  # LTP Check
    '''
    
    # 4. Build and Connect the Synapses
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, 
                   on_pre=on_pre_eqs, on_post=on_post_eqs)
                   
    # Connect Pair 1 (Index 0 connects to Index 1) -> This is syn.w[0]
    syn.connect(i=0, j=1) 
    # Connect Pair 2 (Index 2 connects to Index 3) -> This is syn.w[1]
    syn.connect(i=2, j=3) 
    
    syn.w = 5.0 # Starting baseline weight for both synapses
    
    syn_monitor = StateMonitor(syn, 'w', record=True)
    net = Network(neurons.group, neurons.state_monitor, neurons.spike_monitor, syn, syn_monitor)

    # ---------------------------------------------------------
    # THE EXPERIMENT TIMELINE
    # ---------------------------------------------------------
    net.run(20*ms) # Let neurons settle at resting potential
    
    # --- PAIR 1: LTP PROTOCOL (Pre fires BEFORE Post) ---
    print("Executing LTP Protocol on Pair 1...")
    neurons.group.I[0] = 50.0  # Fire Pre (0)
    net.run(2*ms)
    neurons.group.I[0] = 0.0 
    
    net.run(8*ms) # Wait 8ms (total 10ms gap)
    
    neurons.group.I[1] = 50.0  # Fire Post (1)
    net.run(2*ms)
    neurons.group.I[1] = 0.0 
    
    net.run(30*ms) # Let the network rest
    
    # --- PAIR 2: LTD PROTOCOL (Post fires BEFORE Pre) ---
    print("Executing LTD Protocol on Pair 2...")
    neurons.group.I[3] = 50.0  # Fire Post (3) First!
    net.run(2*ms)
    neurons.group.I[3] = 0.0 
    
    net.run(8*ms) # Wait 8ms (total 10ms gap)
    
    neurons.group.I[2] = 50.0  # Fire Pre (2) Second!
    net.run(2*ms)
    neurons.group.I[2] = 0.0 
    
    net.run(30*ms) # Let it settle

    # ---------------------------------------------------------
    # Plotting the Results
    # ---------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Top Graph: Pair 1 (LTP)
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[0], label='Pre-Synaptic (Fired First)', color='black')
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[1], label='Post-Synaptic (Fired Second)', color='blue')
    ax1.set_title('Pair 1: LTP Protocol (Pre before Post)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend(loc='upper right')
    
    # Middle Graph: Pair 2 (LTD)
    ax2.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[2], label='Pre-Synaptic (Fired Second)', color='black', linestyle='--')
    ax2.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[3], label='Post-Synaptic (Fired First)', color='red')
    ax2.set_title('Pair 2: LTD Protocol (Post before Pre)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.legend(loc='upper right')
    
    # Bottom Graph: Synaptic Weights
    ax3.plot(syn_monitor.t/ms, syn_monitor.w[0], color='blue', linewidth=3, label='Pair 1 Synapse (Strengthened)')
    ax3.plot(syn_monitor.t/ms, syn_monitor.w[1], color='red', linewidth=3, label='Pair 2 Synapse (Weakened)')
    ax3.set_title('Synaptic Weight Changes ($w$)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Connection Strength ($w$)')
    ax3.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig("CA1_Full_STDP_Validation.png", dpi=300)
    print("Saved as 'CA1_Full_STDP_Validation.png'!")

if __name__ == '__main__':
    run_stdp_learning_suite()