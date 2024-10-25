import os
import sys
import numpy as np
import time
from plot_spectra import plot_signal_one_dim
import unit_conversions


w_pu = 4.29 # Energy of the pump pulse in eV
resolution = 300
W_PR = np.linspace(1.0, 5.5, resolution)
TAU = [5] # Width of the envelope in fs
TRAJ = [300] # Number of trajectories, can also be list of int
TSTEP = 201 # Number of time steps in dynamics simulation
SE_TRIGGER = 'yes' # get stimulated emission
ESA_TRIGGER = 'no' # get excited state absorption
GSB_TRIGGER = 'no' # get ground-state bleach
num_states = 31 # total number of states (GS+excited states)
num_states_I = 8 # total number of states in manifold 0+I (GS= excited state)

def envelope(w, tau):
    """
    Function to predict the intensity of the signal using a Gaussian envelope.
    Parameters
    ----------
    w : float
        Energy of the probe pulse in atomic units
    tau : float
        Width of the envelope in atomic units

    Returns
    -------
    Intensity of the signal
    """
    E = np.exp(-(w * tau) ** 2 / 4.) * tau
    return E

def signal(eV, Ha, Ha_0, f, fs, delta=False):
    """
    Function that predicts the intensity of a signal with respect to an envelope or a delta function.
    Parameters
    ----------
    eV : float
        Energy of the probe pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    fs : float
        Width of the laser pulse envelope
    delta :  bool, optional
        Switch to pulse width of 0 fs

    Returns
    -------
    Intensity of the signal
    """
    laser = unit_conversions.eV2Ha(eV)
    tau = unit_conversions.fs2aut(fs)
    if f < 0:
        f=0
    w_in = Ha - Ha_0
    if w_in == 0.0:
        print('Warning:', w_in)
        time.sleep(100)
    dw = laser - w_in
    v2 = f / 2. * 3 / w_in
    if not delta:
        return envelope(dw, tau) ** 2. * v2
    else:
        return 1. * v2

def signal_sum_deV(eV, deV, f, fs):
    """
    Function that predicts the intensity of a signal with respect to an envelope or a delta function for excited state absorption.
    Parameters
    ----------
    eV : float
        Energy of the probe pulse in eV
    deV : list, float
        Energy-differences of states higher than the populated state
    f : float
        Oscillator strength for the transition between the populated state and the higher lying states
    fs : float
        Width of the laser pulse envelope

    Returns
    -------
    Intensity of the signal
    """
    laser = unit_conversions.eV2Ha(eV)
    tau = unit_conversions.fs2aut(fs)
    S = 0
    for (i, val) in enumerate(deV):
        w_in = -1. * unit_conversions.eV2Ha(val)
        dw = laser - w_in
        if w_in != 0:
            v2 = f[i] / 2. * 3 / w_in
            S += envelope(dw, tau) ** 2. * v2
    return S

def signal_sum(eV, Ha, Ha_0, f, fs, delta=False):
    """

    Parameters
    ----------
    eV : float
        Energy of the probe pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    fs : float
        Width of the laser pulse envelope
    delta :  bool, optional
        Switch to pulse width of 0 fs

    Returns
    -------

    """
    laser = unit_conversions.eV2Ha(eV)
    tau = unit_conversions.fs2aut(fs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_in = val - Ha_0
        dw = laser - w_in
        v2 = f[i] / 2. * 3 / w_in
        if not delta:
            S += envelope(dw, tau) ** 2. * v2
        else:
            S += 1. * v2
    return S


print(str(w_pu) + 'eV as pump pulse')
eaEmax = 100
eaEmin = 0
list_intensities = []
for tau in TAU:
    for TRAJNO in TRAJ:
        gsCount = np.zeros([TSTEP, 2])
        exCount = np.zeros([TSTEP, 2])
        eaCount = np.zeros([TSTEP, 2])
        exTCount = gsTCount = eaTCount = 0
        for itraj in range(TRAJNO):
            trajNo = itraj + 1
            exfile = './ex/TRAJ' + str(trajNo) + '/Results/chr_energy.dat' #excited state dynamics output
            exfile_osc = './ex/TRAJ' + str(trajNo) + '/Results/chr_oscill.dat'
            if SE_TRIGGER == 'yes':
                exE = np.zeros([TSTEP, num_states+2]) #t, S0-Sn, current
                exf = np.zeros([TSTEP, num_states]) #t, f1-fn
                for t in range(TSTEP):
                    exCount[t, 0] = exE[t, 0] = exf[t, 0] = t * .5
                if os.path.isfile(exfile):
                    exTCount += 1
                    print('Excited State Dynamics:', 'tau =', tau, ',total TRAJ =', TRAJNO, ',traj =', trajNo, ',exTCount ->', exTCount)
                    with open(exfile, 'r') as f:
                        tstep = 0
                        for line in f:
                            exCount[tstep, 1] += 1
                            exE[tstep, -1] = line.split()[1]
                            exE[tstep, 1:int(len(line.split()[4:])+1)] = line.split()[4:]
                            tstep += 1
                            if tstep == TSTEP:
                                break
                    with open(exfile_osc, 'r') as f:
                        tstep = 0
                        for line in f:
                            exf[tstep, 1:int(len(line.split())+1)] = line.split()
                            tstep += 1
                            if tstep == TSTEP:
                                break
                    Dee = signal(w_pu, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1])-1], tau)
                    Dee_delta = signal(w_pu, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1])-1], tau, delta=True)
                    if ESA_TRIGGER == 'yes':
                        eaE = np.zeros([TSTEP, num_states-1])  # S1-SXX  energy differences between current and  higher lying states in eV
                        eaf = np.zeros([TSTEP, num_states])  # t, df5-df30  osc str between current and higher lying states
                        for t in range(TSTEP):
                            #eaCount[t, 0] = eaE[t, 0] = eaf[t, 0] = t * .5 ### ORIGINAL
                            eaCount[t, 0] = eaf[t, 0] = t * .5
                        eafile = './ex/TRAJ' + str(trajNo) + '/Results/chr_oscillator_strengths_ex.dat'
                        if os.path.isfile(eafile):
                            eaTCount += 1
                            print('Excited State Absorption:', 'tau =', tau, ',total TRAJ =', TRAJNO, ',traj =', trajNo,',eaTCount +>', eaTCount)
                            with open(eafile, 'r') as f:
                                eatstep = 0
                                for line in f:
                                    if eatstep == TSTEP or int(exE[eatstep, -1]) == 0:
                                        break
                                    else:
                                        #Determine which osc. str. to read, states below populated state will stay with f=0
                                        start_index = -num_states + 1### Allow for adjustement in same loop
                                        end_index = 0
                                        for i in range(1, int(exE[eatstep, -1])): ### S1 starts at position 0
                                            start_index += (num_states - i)
                                            end_index += (num_states - i - 1)
                                        eaf[eatstep, int(exE[eatstep, -1]):] = line.split()[start_index:end_index]
                                        eaE[eatstep, int(exE[eatstep, -1]-2):] = -1*unit_conversions.Ha2eV(exE[eatstep, int(exE[eatstep, -1]):-1] - exE[eatstep, int(exE[eatstep, -1])])
                                    eaCount[eatstep, 1] += 1
                                    eatstep += 1
                            for (eai, eaival) in enumerate(eaE[:, 1]):
                                if eaival != 0:
                                    eaCount[eai, 1] += 1
                            for eai in range(len(eaE[:, 0])):
                                if eaEmin < np.amin(-1 * eaE[eai, :]) and np.amin(-1 * eaE[eai, :]) > 0:
                                    eaEmin = np.amin(-1 * eaE[eai, :])
                                if eaEmax > np.amax(-1 * eaE[eai, :]) and np.amax(-1 * eaE[eai, :]) > 0:
                                    eaEmax = np.amax(-1 * eaE[eai, :])
                        eaf[:,-3:] = 0
                        for (iw_pr, w_pr) in enumerate(W_PR):
                            EAi = np.zeros([TSTEP, 3])  # t, w_pr, EA
                            EAi_delta = np.zeros([TSTEP, 3])  # t, w_pr, EA
                            for t in range(TSTEP):
                                EAi[t, 0] = t * .5
                                EAi[t, 1] = w_pr
                                EAi_delta[t, 0] = t * .5
                                EAi_delta[t, 1] = w_pr
                                if t <= eatstep:
                                    if int(exE[t, -1]) != 0:
                                        Wea = signal_sum_deV(w_pr, eaE[t, :], eaf[t, 1:], tau)
                                    else:
                                        Wea = 0
                                else:
                                    Wea = 0
                                EAi[t, 2] = -1. * Dee * Wea
                                EAi_delta[t, 2] = -1. * Dee_delta * Wea
                            if iw_pr == 0:
                                EA = EAi.copy()
                                EA_delta = EAi_delta.copy()
                            else:
                                EA = np.vstack((EA, EAi))
                                EA_delta = np.vstack((EA_delta, EAi_delta))
                        if trajNo == 1:
                            EAsum = EA.copy()
                            EAsum_delta = EA_delta.copy()
                        else:
                            EAsum[:, 2] += EA[:, 2]
                            EAsum_delta[:, 2] += EA_delta[:, 2]
                #window
                for (iw_pr, w_pr) in enumerate(W_PR):
                    SEi = np.zeros([TSTEP, 3]) #t, w_pr, SE
                    SEi_delta = np.zeros([TSTEP, 3]) #t, w_pr, SE
                    for t in range(TSTEP):
                        SEi[t, 0] = t * .5
                        SEi[t, 1] = w_pr
                        SEi_delta[t, 0] = t * .5
                        SEi_delta[t, 1] = w_pr
                        if t < tstep:
                            if int(exE[t, -1]) != 0:
                                Wee = signal(w_pr, exE[t, int(exE[t, -1])], exE[t, 1], exf[t, int(exE[t, -1])-1], tau)
                            else:
                                Wee = 0
                        else:
                            Wee = 0
                        SEi[t, 2] = Dee * Wee
                        SEi_delta[t, 2] = Dee_delta * Wee
                    if iw_pr == 0:
                        SE = SEi.copy()
                        SE_delta = SEi_delta.copy()
                    else:
                        SE = np.vstack((SE, SEi))
                        SE_delta = np.vstack((SE_delta, SEi_delta))
                if trajNo == 1:
                    SEsum = SE.copy()
                    SEsum_delta = SE_delta.copy()
                else:
                    SEsum[:, 2] += SE[:, 2]
                    SEsum_delta[:, 2] += SE_delta[:, 2]
            if GSB_TRIGGER == 'yes':
                gsE = np.zeros([TSTEP, num_states_I+1])    # t, S0-Sn, states in manifold 0+I
                gsf = np.zeros([TSTEP, num_states_I])  # t, f1-fn
                for t in range(TSTEP):
                    gsCount[t, 0] = gsE[t, 0] = gsf[t, 0] = t * .5
                gsfile_energies= 'gsb/TRAJ' + str(trajNo) + '/Results/energy.dat'
                gsfile_oscill= 'gsb/TRAJ' + str(trajNo) + '/Results/oscill.dat'
                if os.path.isfile(gsfile_energies):
                    gsTCount += 1
                    print
                    'Ground State Dynamics:', 'tau =', tau, ',total TRAJ =', TRAJNO, ',traj =', trajNo, ',gsTCount =>', gsTCount
                    with open(gsfile_energies, 'r') as f:
                        tstep = 0
                        next(f)
                        for line in f:
                            gsCount[tstep, 1] += 1
                            gsE[tstep, 1] += 1
                            gsE[tstep, 1:] = line.split()[-num_states_I:]
                            tstep += 1
                            if tstep == TSTEP:
                                break
                    with open(gsfile_oscill, 'r') as f:
                        tstep = 0
                        for line in f:
                            gsf[tstep, 1:num_states_I] = line.split()
                            tstep += 1
                            if tstep == TSTEP:
                                break
                    # doorway
                    Dgg = signal_sum(w_pu, gsE[0, 2:], gsE[0, 1], gsf[0, 1:], tau)
                    Dgg_delta = signal_sum(w_pu, gsE[0, 2:], gsE[0, 1], gsf[0, 1:], tau, delta=True)
                    # window
                    for (iw_pr, w_pr) in enumerate(W_PR):
                        GSBi = np.zeros([TSTEP, 3])  # t, w_pr, GSB
                        GSBi_delta = np.zeros([TSTEP, 3])  # t, w_pr, GSB
                        for t in range(tstep):
                            Wgg = signal_sum(w_pr, gsE[t, 2:], gsE[t, 1], gsf[t, 1:], tau)
                            GSBi[t, 0] = t * .5
                            GSBi[t, 1] = w_pr
                            GSBi_delta[t, 0] = t * .5
                            GSBi_delta[t, 1] = w_pr
                            GSBi[t, 2] = Dgg * Wgg
                            GSBi_delta[t, 2] = Dgg_delta * Wgg
                        if iw_pr == 0:
                            GSB = GSBi.copy()
                            GSB_delta = GSBi_delta.copy()
                        else:
                            GSB = np.vstack((GSB, GSBi))
                            GSB_delta = np.vstack((GSB_delta, GSBi_delta))
                    if trajNo == 1:
                        GSBsum = GSB.copy()
                        GSBsum_delta = GSB_delta.copy()
                    else:
                        GSBsum[:, 2] += GSB[:, 2]
                        GSBsum_delta[:, 2] += GSB_delta[:, 2]
        #normalization
        #t, E, SE/n, GSB/n, EA/n, S/n
        #0, 1,    2,     3,    4,   5
        if SE_TRIGGER == 'yes':
            for i in range(len(SEsum[:,2])):
                SEsum[i, 2] /= exCount[i%TSTEP, -1]
                SEsum_delta[i, 2] /= exCount[i%TSTEP, -1]
            S = SEsum.copy()
            S_delta = SEsum_delta.copy()
            list_intensities += [SEsum[:,2]/max(SEsum[:,2])]
            plot_signal_one_dim(exCount[:,0], W_PR, SEsum[:,2]/max(SEsum[:, 2]), vmax=0.1, vmin=0.0, outname='tr_se_tau' + str(tau) + '_traj' + str(TRAJNO) + '.png')
        if GSB_TRIGGER == 'yes':
            for i in range(len(GSBsum[:, 2])):
                GSBsum[i, 2] /= gsCount[i % TSTEP, -1]
                GSBsum_delta[i, 2] /= gsCount[i % TSTEP, -1]
                GSBsum[i, 2] /= 10
                GSBsum_delta[i, 2] /= 10
            if S.size != 0:
                S = np.hstack((S, GSBsum[:, 2:3]))
                S_delta = np.hstack((S_delta, GSBsum_delta[:, 2:3]))
            else:
                S = GSBsum.copy()
                S_delta = GSBsum_delta.copy()
            plot_signal_one_dim(exCount[:,0], W_PR, GSBsum[:, 2]/max(GSBsum[:, 2]), vmax=1.0, vmin=0.0, outname='tr_gsb_tau' + str(tau) + '_traj' + str(TRAJNO) + '.png')
        if ESA_TRIGGER == 'yes':
            for i in range(len(EAsum[:, 2])):
                EAsum[np.isnan(EAsum)] = 0
                EAsum[i, 2] /= eaCount[i % TSTEP, -1]
                EAsum_delta[i, 2] /= eaCount[i % TSTEP, -1]
            if S.size != 0:
                S = np.hstack((S, EAsum[:, 2:3]))
                S_delta = np.hstack((S_delta, EAsum_delta[:, 2:3]))
            else:
                S = EAsum.copy()
                S_delta = EAsum_delta.copy()
            plot_signal_one_dim(exCount[:, 0], W_PR, -1 * EAsum[:, 2] / min(EAsum[:, 2]), vmax=0.0, vmin=-1.0, outname='tr_esa_tau' + str(tau) + '_traj' + str(TRAJNO) + '.png')

        S = np.hstack((S, S[:, 2:3] + S[:, 3:4] + S[:, 4:5]))
        S_delta = np.hstack((S_delta, S_delta[:, 2:3] + S_delta[:, 3:4] + S_delta[:, 4:5]))
        plot_signal_one_dim(exCount[:, 0], W_PR, S[:, -1]/max(np.absolute(S[:, -1])), vmax=0.4, vmin=-0.4, outname='tr_full_tau' + str(tau) + '_traj' + str(TRAJNO) + '.png')
        with open('int_tau' + str(tau) + '_traj' + str(TRAJNO) + '.dat', 'w') as output:
            np.savetxt(output, S)
        with open('int_delta_tau' + str(tau) + '_traj' + str(TRAJNO) + '.dat', 'w') as output:
            np.savetxt(output, S_delta)

