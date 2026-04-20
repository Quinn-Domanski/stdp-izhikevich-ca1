import matplotlib.pyplot as plt
from brian2 import *
from model import IzhikevichNeuron

def run_emergent_stdp():
    print("Running Emergent STDP (Natural Spiking)...")
    start_scope()
    
    # We need 10 neurons total.
    # LTP Group: Indices 0,1,2,3 (Pre) connected to 4 (Post)
    # LTD Group: Indices 5,6,7,8 (Pre) connected to 9 (Post)
    neurons = IzhikevichNeuron(N=10, b=0.2347, c=-66.5)

    # STDP Parameters
    tau_pre = 20*ms    
    tau_post = 20*ms   
    A_pre = 2.0        # LTP Jump
    A_post = -2.0      # LTD Drop
    w_max = 10.0       
    w_min = 0.0        
    
    # Synapse Math
    syn_eqs = '''
    w : 1
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    on_pre_eqs = '''
    v_post += w                
    apre += A_pre              
    w = clip(w + apost, w_min, w_max) 
    '''
    on_post_eqs = '''
    apost += A_post            
    w = clip(w + apre, w_min, w_max)  
    '''
    
    syn = Synapses(neurons.group, neurons.group, model=syn_eqs, 
                   on_pre=on_pre_eqs, on_post=on_post_eqs)
                   
    # Connect the LTP Group (4 Pre to 1 Post)
    syn.connect(i=[0, 1, 2, 3], j=4) 
    # Connect the LTD Group (4 Pre to 1 Post)
    syn.connect(i=[5, 6, 7, 8], j=9) 
    
    # Set initial weights to 4.5. 
    # 4 neurons * 4.5 = 18mV jump. This is just enough to cross the -50mV threshold!
    syn.w = 4.5 
    
    # Monitor one representative synapse from each group
    syn_monitor = StateMonitor(syn, 'w', record=[0, 4]) # 0 is LTP, 4 is LTD
    net = Network(neurons.group, neurons.state_monitor, neurons.spike_monitor, syn, syn_monitor)

    net.run(20*ms)
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: NATURAL LTP
    # We only stimulate the 4 Pre-neurons. The Post-neuron fires ON ITS OWN.
    # ---------------------------------------------------------
    print("Triggering Natural LTP...")
    neurons.group.I[0:4] = 50.0  
    net.run(2*ms)
    neurons.group.I[0:4] = 0.0 
    
    net.run(40*ms) # Wait and watch the Post-neuron naturally spike!
    
    # ---------------------------------------------------------
    # EXPERIMENT 2: FORCED LTD
    # We force the Post-neuron to fire, then fire the Pre-neurons late.
    # ---------------------------------------------------------
    print("Triggering LTD...")
    neurons.group.I[9] = 50.0  # Force Post-neuron to fire first
    net.run(2*ms)
    neurons.group.I[9] = 0.0 
    
    net.run(5*ms) # Wait 5ms
    
    neurons.group.I[5:9] = 50.0  # Fire Pre-neurons late
    net.run(2*ms)
    neurons.group.I[5:9] = 0.0 
    
    net.run(40*ms)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # LTP Graph
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[0], label='Pre-Neurons (Summating)', color='black')
    ax1.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[4], label='Post-Neuron (Natural Spike)', color='blue')
    ax1.set_title('Natural LTP: Spatial Summation causes Emergent Spike')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend()
    
    # LTD Graph
    ax2.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[5], label='Pre-Neurons (Late)', color='black', linestyle='--')
    ax2.plot(neurons.state_monitor.t/ms, neurons.state_monitor.v[9], label='Post-Neuron (Forced Early)', color='red')
    ax2.set_title('LTD Protocol: Anti-Causal Spiking')
    ax2.set_ylabel('Voltage (mV)')
    ax2.legend()
    
    # Weight Graph
    ax3.plot(syn_monitor.t/ms, syn_monitor.w[0], color='blue', linewidth=3, label='LTP Synapse Weight')
    ax3.plot(syn_monitor.t/ms, syn_monitor.w[1], color='red', linewidth=3, label='LTD Synapse Weight')
    ax3.set_title('Synaptic Plasticity (Weight Changes)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Connection Strength (w)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("CA1_Emergent_STDP.png", dpi=300)
    print("Saved as 'CA1_Emergent_STDP.png'!")

if __name__ == '__main__':
    run_emergent_stdp()