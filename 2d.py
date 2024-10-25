import os, sys
import numpy as np
from plot_spectra import plot_signal_two_dim
import unit_conversions


if len(sys.argv) <= 1:
    print("enter Tw as 1st argument")
    exit()
tw = int(sys.argv[1]) 

NIU = [0.1] # Dephasing in eV
w_pu = 4.37 # Energy of the pump pulse in eV
w_pr = 4.37 # Energy of the probe pulse in eV
resolution = 300
W_T = np.linspace(1.0,
                   6.0,
                   resolution)
TAU = [0.1] # Width of the envelope in fs
TRAJ = [150] # Number of trajectories, can also be list of int
TSTEP = 201 # Number of time steps in dynamics simulation
SE_TRIGGER = 'yes' # get stimulated emission
ESA_TRIGGER = 'yes' # get excited state absorption contribution
GSB_TRIGGER = 'yes' # get ground-state bleach

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
    Function that predicts the intensity of a signal with respect to an envelope or a delta function, Doorway
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
    if f<0:
        f=0
    w_in = Ha - Ha_0
    dw = laser - w_in
    v2 = f / 2. * 3 / w_in
    if not delta:
        return envelope(dw, tau) ** 2. * v2
    else:
        return 1. * v2

def dissignal(w_in, Ha, Ha_0, f, taufs, niueV, w_preV): # input w=eV and tau=fs
    """
    Function that predicts the intensity of a dissipation signal with respect to an envelope or a delta function, Doorway

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV

    Returns
    -------
    Intensity of the dissipation signal
    """
    w_t = unit_conversions.eV2Ha(w_in)
    w_pr = unit_conversions.eV2Ha(w_preV)
    niu = unit_conversions.eV2Ha(niueV)
    tau = unit_conversions.fs2aut(taufs)
    w_ge = Ha - Ha_0
    dw = w_t - w_ge
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 * niu / (niu ** 2 + dw ** 2)

def dissignal_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV):
    """
    Function that predicts the intensity of a rephasing dissipation signal with respect to an envelope or a delta function, Doorway

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV

    Returns
    -------
    Intensity of the rephasing dissipation signal
    """
    w_t = unit_conversions.eV2Ha(w_in)       #parameter frequency
    w_pr = unit_conversions.eV2Ha(w_preV)    #fixed frequency
    niu = unit_conversions.eV2Ha(niueV)      #broadening
    tau = unit_conversions.fs2aut(taufs)      #Tw
    w_ge = Ha - Ha_0        #Ueg
    dw = w_t - w_ge         #dispersion
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, dw)

def dissignal_NR(w_in, Ha, Ha_0, f, taufs, niueV, w_preV):
    """
    Function that predicts the intensity of a non-rephasing dissipation signal with respect to an envelope or a delta function, Doorway

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV

    Returns
    -------
    Intensity of the non-rephasing dissipation signal
    """
    w_t = unit_conversions.eV2Ha(w_in)
    w_pr = unit_conversions.eV2Ha(w_preV)
    niu = unit_conversions.eV2Ha(niueV)
    tau = unit_conversions.fs2aut(taufs)
    w_ge = Ha - Ha_0
    dw = w_t - w_ge
    v2 = f / 2. * 3 / w_ge
    return envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)

def dissignal_sum_NR_ESA(w_in, dEeV, f, taufs, niueV, w_preV):
    """

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    dEeV : list, float
        Energy-differences between current and higher lying states
    f : list, float
        Oscillator strength between current and other excited states
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV

    Returns
    -------

    """
    w_t = unit_conversions.eV2Ha(w_in)
    w_pr = unit_conversions.eV2Ha(w_preV)
    niu = unit_conversions.eV2Ha(niueV)
    tau = unit_conversions.fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(dEeV):
        w_ge = -1. * unit_conversions.eV2Ha(val)
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)
    return S

def dissignal_sum_R(w_in, Ha, Ha_0, f, taufs, niueV, w_preV):
    """

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV
    Returns
    -------

    """
    w_t = unit_conversions.eV2Ha(w_in)
    w_pr = unit_conversions.eV2Ha(w_preV)
    niu = unit_conversions.eV2Ha(niueV)
    tau = unit_conversions.fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, dw)
    return S

def dissignal_sum_NR(w_in, Ha, Ha_0, f, taufs, niueV, w_preV):
    """

    Parameters
    ----------
    w_in : float
        Energy of the pump pulse in eV
    Ha : float
        Energy of the populated state in atomic units
    Ha_0 : float
        Energy of the electronic ground state in atomic units
    f :  float
        Oscillator strength for the transition between the populated state and the electronic ground state
    taufs : float
        Width of the envelope in femto-seconds
    niueV : float
        Dephasing in eV
    w_preV : float
        Energy of the probe pulse in eV

    Returns
    -------

    """
    w_t = unit_conversions.eV2Ha(w_in)
    w_pr = unit_conversions.eV2Ha(w_preV)
    niu = unit_conversions.eV2Ha(niueV)
    tau = unit_conversions.fs2aut(taufs)
    S = 0
    for (i, val) in enumerate(Ha):
        w_ge = val - Ha_0
        dw = w_t - w_ge
        if w_ge != 0:
            v2 = f[i] / 2. * 3 / w_ge
            S += envelope(w_t - w_pr, tau) ** 2. * v2 / complex(niu, -dw)
    return S


print(str(w_pu) + ' eV as pump frequency')
print(str(w_pr) + ' eV as probe frequency')
eaEmax = 100
eaEmin = 0
print(str(tw) + ' fs as Tw')
tws = tw * 2

GSC = [150]
list_intensities = []

for gsC in GSC:
  for niu in NIU:
    for tau in TAU:
        for TRAJNO in TRAJ:
            gsCount, exCount, eaCount = [np.zeros([TSTEP, 2], dtype=int) for _ in range(3)]
            exTCount = gsTCount = eaTCount = 0
            for itraj in range(TRAJNO):
                trajNo = itraj + 1
                gsfile_energies= 'gsb/TRAJ' + str(trajNo) + '/Results/energy.dat'
                gsfile_oscill= 'gsb/TRAJ' + str(trajNo) + '/Results/oscill.dat'
                exfile = './ex/TRAJ' + str(trajNo) + '/Results/chr_energy.dat' # energies of excited states in a.u.
                exfile_osc = './ex/TRAJ' + str(trajNo) + '/Results/chr_oscill.dat' # osc str between GS and excited states
                eafile = './ex/TRAJ' + str(trajNo) + '/Results/chr_oscillator_strengths_ex.dat' # osc str between excited states
                gscheck = False
                excheck = False
                eacheck = False
                if exCount[tws, 1] >= gsC:
                    break
                ############################
                #  read ground state dyn   #
                ############################
                if GSB_TRIGGER == 'yes':
                    if os.path.isfile(gsfile_energies):
                        gscheck = True
                        gsTCount += 1
                        gsE = np.zeros([TSTEP, num_states_I + 1])  # t, S0-Sn, states in manifold 0+I
                        gsf = np.zeros([TSTEP, num_states_I])  # t, f1-fn
                        for t in range(TSTEP):
                            gsCount[t, 0] = gsE[t, 0] = gsf[t, 0] = t * .5
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
                ############################
                #  read excited dynamics   #
                ############################
                if SE_TRIGGER == 'yes':
                    if os.path.isfile(exfile):
                        excheck = True
                        exTCount += 1
                        exE = np.zeros([TSTEP, num_states + 2])  # t, S0-Sn, current
                        exf = np.zeros([TSTEP, num_states])  # t, f1-fn
                        for t in range(TSTEP):
                            exCount[t, 0] = exE[t, 0] = exf[t, 0] = t * .5
                        with open(exfile,'r') as f:
                            tstep = 0
                            for line in f:
                                exCount[tstep, 1] += 1
                                exE[tstep, -1] = line.split()[1]
                                exE[tstep, 1:-1] = line.split()[-num_states:]
                                tstep += 1
                                if tstep == TSTEP:
                                    break
                        with open(exfile_osc,'r') as f:
                            tstep = 0
                            for line in f:
                                exf[tstep, 1:num_states] = line.split()
                                tstep += 1
                                if tstep == TSTEP:
                                    break
                    ############################
                    #  read ESA dynamics       #
                    ############################
                        if ESA_TRIGGER == 'yes':
                            if os.path.isfile(eafile):
                                eacheck = True
                                eaTCount += 1
                                eaE = np.zeros([TSTEP,num_states - 1])  # t, S1-SXX  energy differences between current and  higher lying states in eV
                                eaf = np.zeros([TSTEP, num_states])  # t,  osc str between current and higher lying states
                                for t in range(TSTEP):
                                    eaCount[t, 0] = eaE[t, 0] = eaf[t, 0] = t * .5
                                with open(eafile, 'r') as f:
                                    eatstep = 0
                                    for line in f:
                                        if eatstep == TSTEP or int(exE[eatstep, -1]) == 0: ### just in case a line is empty
                                            break
                                        else:
                                            # Determine which osc. str. to read, states below populated state will stay with f=0
                                            start_index = -num_states + 1  ### Allow for adjustement in same loop
                                            end_index = 0
                                            for i in range(1, int(exE[eatstep, -1])):  ### S1 starts at position 0
                                                start_index += (num_states - i)
                                                end_index += (num_states - i - 1)
                                            eaf[eatstep, int(exE[eatstep, -1]):] = line.split()[start_index:end_index]

                                            eaE[eatstep, int(exE[eatstep, -1] - 2):] = -1 * unit_conversions.Ha2eV(exE[eatstep, int(exE[eatstep, -1]):-1] - exE[eatstep, int(exE[eatstep, -1])])
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


                ############################
                #  calculating DW          #
                ############################
                eaf[:, -3:] = 0
                for (iw_t, w_t) in enumerate(W_T):
                    GSB_R, GSB_NR = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity, Mean Intensity
                    SE_R, SE_NR = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity
                    EA_NR, EA_R = [np.zeros([resolution, 3]) for _ in range(2)] #w_tau, w_t, Intensity
                    for (iw_tau, w_tau) in enumerate(W_T):
                        if gscheck:
                            GSB_R[iw_tau, 0] = GSB_NR[iw_tau, 0] = w_tau
                            Dgg_R = dissignal_sum_R(w_tau, gsE[0, 2:], gsE[0, 1], gsf[0, 1:], tau, niu, w_pu)
                            Dgg_NR = dissignal_sum_NR(w_tau, gsE[0, 2:], gsE[0, 1], gsf[0, 1:], tau, niu, w_pu)
                            Wgg = dissignal_sum_NR(w_t, gsE[tws, 2:], gsE[tws, 1], gsf[tws, 1:], tau, niu, w_pr)
                            GSB_R[iw_tau, 2] = np.real(Dgg_R * Wgg)
                            GSB_NR[iw_tau, 2] = np.real(Dgg_NR * Wgg)
                        if excheck:
                            SE_R[iw_tau, 0] = SE_NR[iw_tau, 0] = w_tau
                            Dee_R = dissignal_R(w_tau, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1]) - 1], tau, niu, w_pu)
                            Dee_NR = dissignal_NR(w_tau, exE[0, int(exE[0, -1])], exE[0, 1], exf[0, int(exE[0, -1]) - 1], tau, niu, w_pu)
                            if int(exE[tws, -1]) != 0:
                                Wee = dissignal_NR(w_t, exE[tws, int(exE[tws, -1])], exE[tws, 1], exf[tws, int(exE[tws, -1]) - 1], tau, niu, w_pr)
                            else:
                                Wee = 0
                            SE_R[iw_tau, 2] = np.real(Dee_R * Wee)
                            SE_NR[iw_tau, 2] = np.real(Dee_NR * Wee)
                            if eacheck:
                                EA_NR[iw_tau, 0] = EA_R[iw_tau, 0] = w_tau
                                if int(exE[tws, -1]) != 0:
                                    #if eaE[tws, 1] == 0:
                                    #    Wea = 0
                                    #else:

                                    Wea = dissignal_sum_NR_ESA(w_t, eaE[tws, int(exE[tws, -1]):], eaf[tws, int(exE[tws, -1]):], tau, niu,w_pr)

                                else:
                                    Wea = 0
                                EA_R[iw_tau, 2] = -1. * np.real(Dee_R * Wea)
                                EA_NR[iw_tau, 2] = -1. * np.real(Dee_NR * Wea)
                    if gscheck:
                        GSB_R[:, 1] = GSB_NR[:, 1] = w_t
                        if iw_t == 0:
                            GSB_R_full = GSB_R.copy()
                            GSB_NR_full = GSB_NR.copy()
                        else:
                            GSB_R_full = np.vstack((GSB_R_full, GSB_R))
                            GSB_NR_full = np.vstack((GSB_NR_full, GSB_NR))
                    if excheck:
                        SE_R[:, 1] = SE_NR[:, 1] = w_t
                        if iw_t == 0:
                            SE_R_full = SE_R.copy()
                            SE_NR_full = SE_NR.copy()
                        else:
                            SE_R_full = np.vstack((SE_R_full, SE_R))
                            SE_NR_full = np.vstack((SE_NR_full, SE_NR))
                    if eacheck:
                        EA_NR[:, 1] = EA_R[:, 1] = w_t
                        if iw_t == 0:
                            EA_R_full = EA_R.copy()
                            EA_NR_full = EA_NR.copy()
                        else:
                            EA_R_full = np.vstack((EA_R_full, EA_R))
                            EA_NR_full = np.vstack((EA_NR_full, EA_NR))
                #Accumulation of trajectories
                if gscheck:
                    if trajNo == 1:
                        GSB_R_full_sum = GSB_R_full.copy()
                        GSB_NR_full_sum = GSB_NR_full.copy()
                    else:
                        GSB_R_full_sum[:, -1] += GSB_R_full.copy()[:, -1]
                        GSB_NR_full_sum[:, -1] += GSB_NR_full.copy()[:, -1]
                if excheck:
                    if trajNo == 1:
                        SE_R_full_sum = SE_R_full.copy()
                        SE_NR_full_sum = SE_NR_full.copy()
                    else:
                        SE_R_full_sum[:, -1] += SE_R_full.copy()[:, -1]
                        SE_NR_full_sum[:, -1] += SE_NR_full.copy()[:, -1]
                if eacheck:
                    if trajNo == 1:
                        EA_R_full_sum = EA_R_full.copy()
                        EA_NR_full_sum = EA_NR_full.copy()
                    else:
                        EA_R_full_sum[:, -1] += EA_R_full.copy()[:, -1]
                        EA_NR_full_sum[:, -1] += EA_NR_full.copy()[:, -1]
            ###############
            #normalization#
            ###############
            #w_tau, w_t, SE/n, GSB/n, EA/n, S/n
            #    0,   1,    2,     3,    4,   5
            print('Trajectory Count: gs ' + str(gsTCount) + ' ; ex ' + str(exTCount) + ' ; ea ' + str(eaTCount))
            print('Valid points count: at ' + str(gsCount[tws, 0]) + ' fs, gs ' + str(gsCount[tws, 1]) + ' ; ex ' + str(exCount[tws, 1]) + ' ; ea ' + str(eaCount[tws, 1]))
            if SE_TRIGGER == 'yes':
                SE_R = SE_R_full_sum.copy()
                SE_NR = SE_NR_full_sum.copy()
                SE_R[:, -1] /= exCount[tws, 1]
                SE_NR[:, -1] /= exCount[tws, 1]
                plot_signal_two_dim(W_T,SE_NR[:,2]/np.amax(SE_NR[:,2]),vmax=0.8,vmin=-0.2,outname='2d_tr_se_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                plot_signal_two_dim(W_T,SE_R[:,2]/np.amax(SE_R[:,2]),vmax=0.8,vmin=-0.2,outname='2d_tr_se_R_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                plot_signal_two_dim(W_T,(SE_R[:,2] + SE_NR[:,2]) /np.amax(SE_R[:,2] + SE_NR[:,2]),vmax=0.8,vmin=-0.2,outname='2d_tr_se_sum_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                S_R = SE_R.copy()
                S_NR = SE_NR.copy()
            if GSB_TRIGGER == 'yes':
                GSB_R = GSB_R_full_sum.copy()
                GSB_NR = GSB_NR_full_sum.copy()
                GSB_R[:, -1] /= gsCount[tws, 1]
                GSB_NR[:, -1] /= gsCount[tws, 1]
                GSB_R[:, -1] /= 10
                GSB_NR[:, -1] /= 10
                if S_R.size != 0:
                    S_R = np.hstack((S_R, GSB_R[:, 2:3]))
                    S_NR = np.hstack((S_NR, GSB_NR[:, 2:3]))
                else:
                    S_R = GSB_R.copy()
                    S_NR = GSB_NR.copy()
                plot_signal_two_dim(W_T, GSB_NR[:, 2]/np.amax(GSB_NR[:,2]),vmax=1.0,vmin=-0.2,outname='2d_tr_gsb_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                plot_signal_two_dim(W_T, GSB_R[:, 2]/np.amax(GSB_R[:,2]),vmax=1.0,vmin=-0.2,outname='2d_tr_gsb_R_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                plot_signal_two_dim(W_T, (GSB_R[:, 2] + GSB_NR[:, 2])/np.amax(GSB_R[:, 2] + GSB_NR[:, 2]),vmax=1.0,vmin=-0.2,outname='2d_tr_gsb_sum_tau' + str(tau) + '_' + str(tw) + 'fs' + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
            if ESA_TRIGGER == 'yes':
                EA_R = EA_R_full_sum.copy()
                EA_NR = EA_NR_full_sum.copy()
                EA_R[:, -1] /= eaCount[tws, -1]
                EA_NR[:, -1] /= eaCount[tws, -1]
                EA_R[np.isnan(EA_R)] = 0
                EA_NR[np.isnan(EA_NR)] = 0
                if S_R.size != 0:
                    S_R = np.hstack((S_R, EA_R[:, 2:3]))
                    S_NR = np.hstack((S_NR, EA_NR[:, 2:3]))
                else:
                    S_R = EA_R.copy()
                    S_NR = EA_NR.copy()
                plot_signal_two_dim(W_T, -1 * EA_NR[:, 2] / np.amin(EA_NR[:, 2]), vmax=0.3, vmin=-1.0, outname='2d_tr_esa_tau' + str(tau) + '_' + str(tw) + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                plot_signal_two_dim(W_T, -1 * EA_R[:, 2] / np.amin(EA_R[:, 2]), vmax=0.3, vmin=-1.0, outname='2d_tr_esa_R_tau' + str(tau) + '_' + str(tw) + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
                plot_signal_two_dim(W_T, -1 * (EA_R[:, 2] + EA_NR[:, 2])/ np.amin(EA_R[:, 2] + EA_NR[:, 2]), vmax=0.3, vmin=-1.0, outname='2d_tr_esa_sum_tau' + str(tau) + '_' + str(tw) + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
            S_R = np.hstack((S_R, S_R[:, 2:3] + S_R[:, 3:4] + S_R[:, 4:5]))
            S_NR = np.hstack((S_NR, S_NR[:, 2:3] + S_NR[:, 3:4] + S_NR[:, 4:5]))
            S = S_R.copy()
            S[:, 2:6] += S_NR[:, 2:6]
            plot_signal_two_dim(W_T, S_NR[:, -1] / np.amax(np.absolute(S_NR[:, -1])), vmax=0.5, vmin=-1.0, outname='2d_tr_tot_tau' + str(tau) + '_' + str(tw) + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
            plot_signal_two_dim(W_T, S_R[:, -1] / np.amax(np.absolute(S_R[:, -1])), vmax=0.5, vmin=-1.0, outname='2d_tr_tot_R_tau' + str(tau) + '_' + str(tw) + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
            plot_signal_two_dim(W_T, (S_R[:, -1] + S_NR[:, 2]) / np.amax(np.absolute(S_R[:, -1] + S_NR[:, 2])), vmax=0.5, vmin=-1.0, outname='2d_tr_tot_sum_tau' + str(tau) + '_' + str(tw) + '_traj' + str(TRAJNO) + '_nu' + str(niu) + '.png')
            Smax = np.maximum(abs(S_R[:, 2:6]).max(), abs(S_NR[:, 2:6]).max())
            #print(Smax)
            S_R[:, 2:6] /= Smax 
            with open('new2D_adc_R_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S_R)
            S_NR[:, 2:6] /= Smax
            with open('new2D_adc_NR_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S_NR)
            S[:, 2:6] /= Smax
            with open('new2D_adc_S_tw' + str(tw) + '_tau' + str(tau) + '_pr' + str(w_pr) + '_niu' + str(niu) + '_traj' + str(gsC) + '.dat', 'w') as output:
                np.savetxt(output, S)


