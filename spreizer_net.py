# Copyright 2021 Leo Hiselius
# The MIT License

from brian2 import *
from perlin import generate_perlin
import numpy as np


class SpreizerNet:

    def __init__(self, Ne, Ni, connectivity='hetero_gauss', stp=True, simulation_time=1*second,
                 sigma_e=0.075, sigma_i=0.1, tau_recovery=100*ms, mu_gwn=350*pA,
                 grid_offset=1, input_seed=8, report=True,
                 simulation_name='no_name', hi_thres_mode=False):
        '''
        Ne, Ni are the number of neurons in the respective populations
        connectivity is a str specifying what type of connectivity
        ext_input is a str specifying what type of external input
        tau_recovery is a synaptic time constant
        mu_gwn is the mean of the background Gaussian white noise
        the sigmas are parameters for the 2D gaussian connectivity kernel
        grid_offset determines how strongly neurons are spatially correlated
        input_seed sets the seed for the simualtion
        report is a boolean specifying whether or not to report sim progress
        simulation_name is a str which will be present in all the saved figures
        hi_thres_mode is a boolean specifying whether or not
        '''

        assert sqrt(Ne) == int(sqrt(Ne)), 'Ne is not a square number'
        assert sqrt(Ni) == int(sqrt(Ni)), 'Ni is not a square number'
        assert connectivity == 'CUBA' or connectivity == 'homo_gauss' \
            or connectivity == 'hetero_gauss',\
            'connectivity argument "' + connectivity + '" in method SpikingSTPNeurons.__init__ is not available.' \
            + ' Please use one of the following: \n"CUBA"\n"homo_gauss"\n"hetero_gauss"'

        # Basic parameters for the simulation
        self.input_seed = input_seed
        seed(input_seed)
        np.random.seed(input_seed)
        self.Ne = Ne
        self.Ni = Ni
        self.simulation_time = simulation_time

        # Constants. (from space to time table 1 & 3)
        Cm = 250 * pF
        gL = 25 * nsiemens
        taum = Cm / gL
        El = -70 * mV
        if not hi_thres_mode:
            Vt = -55 * mV
        Vr = -70 * mV
        sigma_gwn = 100 * pA
        taue = 5 * ms
        taui = 5 * ms
        tau_inactivation = 3 * ms
        t_ref = 2 * ms
        synapse_delay = 1 * ms
        calibration_factor = 1.07
        p_max_e = calibration_factor * 0.05 / (2 * pi * sigma_e**2)
        p_max_i = 0.05 / (2 * pi * sigma_i**2)
        we = 0.22 * mV
        wi = -1.76 * mV
        frac_e = 0.2
        max_capac_e = we / frac_e

        # Define dynamic equations. x_shift and y_shift are variables for orientation dependent connectivity
        eqs = '''
                dv/dt = (-gL * (v - El - ve - vi) + mu_gwn + ms * sqrt(1/taum) * sigma_gwn * xi) / Cm : volt (unless refractory)
                dve/dt = -ve/taue : volt
                dvi/dt = -vi/taui : volt
                x : 1
                y : 1
                x_shift : 1
                y_shift : 1
                '''
        if hi_thres_mode:
            eqs += 'Vt : volt'

        # Instantiate the excitatory and inhibitory networks
        neural_net_e = NeuronGroup(Ne, eqs, threshold='v>Vt',
                                   reset='v=Vr', refractory=t_ref, method='euler')

        neural_net_i = NeuronGroup(Ni, eqs, threshold='v>Vt',
                                   reset='v=Vr', refractory=t_ref, method='euler')

        # Place neurons on evenly spaced grid [0,1)x[0,1)
        rows_e = sqrt(Ne)
        rows_i = sqrt(Ni)
        neural_net_e.x = '(i // rows_e) / rows_e'
        neural_net_e.y = '(i % rows_e) / rows_e'
        neural_net_i.x = '(i // rows_i) / rows_i'
        neural_net_i.y = '(i % rows_i) / rows_i'

        if hi_thres_mode:

            # Threshold membrane potential
            neural_net_e.Vt = -55*mV
            neural_net_i.Vt = -55*mV

            # neurons which will not spike
            random_e_idcs = np.random.randint(0, Ne, 8)
            random_i_idcs = np.random.randint(0, Ni, 2)

            neural_net_e.Vt[random_e_idcs] = 1*volt
            neural_net_i.Vt[random_i_idcs] = 1*volt

        # initial potentials for all neurons
        # extra variable Vthres needed to set initial potentials when hi_thres_mode is True
        Vthres = -55*mV
        neural_net_e.v = 'Vr + rand() * (Vthres - Vr)'
        neural_net_e.ve = 0 * mV
        neural_net_e.vi = 0 * mV
        neural_net_i.v = 'Vr + rand() * (Vthres - Vr)'
        neural_net_i.ve = 0 * mV
        neural_net_i.vi = 0 * mV

        # Synapse model
        if stp:
            esyn_model = '''
                        dx_syn/dt = z_syn / tau_recovery: volt (clock-driven)
                        dy_syn/dt = - y_syn / tau_inactivation : volt (clock-driven)
                        dz_syn/dt = y_syn / tau_inactivation - z_syn / tau_recovery : volt (clock-driven)
                        '''

            esyn_eqs = '''
                        ve_post += frac_e * x_syn
                        y_syn += frac_e * x_syn
                        x_syn -= frac_e * x_syn
                        '''

            Cee = Synapses(neural_net_e, model=esyn_model,
                           on_pre=esyn_eqs, method='euler')
            Cie = Synapses(neural_net_e, neural_net_i,
                           model=esyn_model, on_pre=esyn_eqs, method='euler')
            Cei = Synapses(neural_net_i, neural_net_e, on_pre='vi += wi')
            Cii = Synapses(neural_net_i, on_pre='vi += wi')

        elif not stp:
            Cee = Synapses(neural_net_e, on_pre='ve += we')
            Cie = Synapses(neural_net_e, neural_net_i, on_pre='ve += we')
            Cei = Synapses(neural_net_i, neural_net_e, on_pre='vi += wi')
            Cii = Synapses(neural_net_i, on_pre='vi += wi')

        # Define toroidal distance
        @implementation('cython', '''
            cdef double torus_distance(double x_pre, double x_post, double y_pre, double y_post):
               x_pre = x_pre % 1
               y_pre = y_pre % 1

               cdef double dx = abs(x_pre - x_post)
               cdef double dy = abs(y_pre - y_post)
               
               if dx > 0.5:
                  dx = 1 - dx
                  
               if dy > 0.5:
                  dy = 1 - dy
                  
               return sqrt(dx*dx + dy*dy)
            ''')
        @check_units(x_pre=1, x_post=1, y_pre=1, y_post=1, result=1)
        def torus_distance(x_pre, x_post, y_pre, y_post):
            x_pre = x_pre % 1
            y_pre = y_pre % 1

            dx = abs(x_pre - x_post)
            dy = abs(y_pre - y_post)

            if dx > 0.5:
                dx = 1 - dx

            if dy > 0.5:
                dy = 1 - dy

            return sqrt(dx * dx + dy * dy)

        # Connect neurons
        if connectivity == 'homo_gauss':
            Cee.connect(
                p='p_max_e*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_e**2))')
            Cie.connect(
                p='p_max_e*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_e**2))')
            Cei.connect(
                p='p_max_i*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_i**2))')
            Cii.connect(
                p='p_max_i*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_i**2))')

        elif connectivity == 'hetero_gauss':
            perlin_noise = generate_perlin(int(sqrt(Ne)), 30,
                                           save=True,
                                           PATH='figures/scale_'+str(30))

            # set the x_shift and y_shift parameters
            idx = 0
            for i in range(int(rows_e)):
                for j in range(int(rows_e)):
                    neural_net_e.x_shift[idx] = grid_offset / \
                        rows_e * np.cos(perlin_noise[i, j])
                    neural_net_e.y_shift[idx] = grid_offset / \
                        rows_e * np.sin(perlin_noise[i, j])
                    idx += 1

            Cee.connect(
                p='p_max_e*exp(-(torus_distance(x_pre+x_shift_pre, x_post, y_pre+y_shift_pre, y_post)**2)/(2*sigma_e**2))')
            Cie.connect(
                p='p_max_e*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_e**2))')
            Cei.connect(
                p='p_max_i*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_i**2))')
            Cii.connect(
                p='p_max_i*exp(-(torus_distance(x_pre, x_post, y_pre, y_post)**2)/(2*sigma_i**2))')

        elif connectivity == 'CUBA':
            Cee.connect(p=0.02)
            Cie.connect(p=0.02)
            Cei.connect(p=0.02)
            Cii.connect(p=0.02)

        if stp:
            Cee.x_syn = max_capac_e
            Cie.x_syn = max_capac_e

        # Synapse delay
        Cee.delay = 'synapse_delay'
        Cie.delay = 'synapse_delay'
        Cei.delay = 'synapse_delay'
        Cii.delay = 'synapse_delay'

        # SpikeMonitors
        spike_mon_e = SpikeMonitor(neural_net_e, record=True)
        spike_mon_i = SpikeMonitor(neural_net_i, record=True)

        # StateMonitors
        if hi_thres_mode:
            state_mon_e_hi = StateMonitor(
                neural_net_e, 'v', record=random_e_idcs)
            state_mon_i_hi = StateMonitor(
                neural_net_i, 'v', record=random_i_idcs)

        # Run the simulation
        duration = simulation_time
        if report:
            run(duration, report='stderr', report_period=500 * ms)
        else:
            run(duration)

        # Save spike_mons
        np.save('spike_mons/'+simulation_name+'_e_i', spike_mon_e.i[:])
        np.save('spike_mons/'+simulation_name+'_e_t', spike_mon_e.t[:])
        np.save('spike_mons/'+simulation_name+'_i_i', spike_mon_i.i[:])
        np.save('spike_mons/'+simulation_name+'_i_t', spike_mon_i.t[:])

        # Save state_mons
        if hi_thres_mode:
            np.save('state_mons/'+simulation_name +
                    '_e_v_hi', state_mon_e_hi.v[:][:])
            np.save('state_mons/'+simulation_name +
                    '_e_t_hi', state_mon_e_hi.t[:])
            np.save('state_mons/'+simulation_name +
                    '_i_v_hi', state_mon_i_hi.v[:][:])
            np.save('state_mons/'+simulation_name +
                    '_i_t_hi', state_mon_i_hi.t[:])
