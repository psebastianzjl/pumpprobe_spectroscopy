import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def plot_signal_one_dim(time, energy, value, vmax = 0.1, vmin = 0.0, outname='spectrum.png'):
    """
    Function that plots the signal.
    Parameters
    ----------
    time : list or array, float
    List or array containing the information about the time.
    energy : list or array, float
    List or array containing the information about the energy
    value : list or array, float
    List or array containing information the signal intensities.
    outname : str, optional
    Name of the outputfile

    Returns
    -------
    PNG-file containing the spectrum.
    """
    value = np.reshape(value, (int(len(value)/len(time)), len(time))).astype(np.float32)
    value = np.nan_to_num(value) 
    time = np.array(time, dtype=float)
    time_new = time
    energy = np.array(energy, dtype=float) 
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(time_new, energy, value, cmap='rainbow', vmin=vmin, vmax=vmax, shading='nearest')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=14) 
    ax.axis([time_new.min(), time_new.max(), energy.min(), energy.max()])
    #ax.axis([time_new.min(), time_new.max(), energy.min(), energy.max()])

    ax.set_xlabel(r'Population Time $T$, [fs]', fontsize=20)
    ax.set_ylabel(r'$\hbar\omega_{pr}$, [eV]', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    #plt.show()
    plt.savefig(outname, bbox_inches='tight',dpi=400)
    plt.close()

def plot_signal_two_dim(energy, value, vmax = 0.3, vmin = -0.3,outname='spectrum.png'):
    """
    Function that plots the signal.
    Parameters
    ----------
    energy : list or array, float
    List or array containing the information about the energy
    value : list or array, float
    List or array containing information the signal intensities.
    outname : str, optional
    Name of the outputfile

    Returns
    -------
    PNG-file containing the spectrum.
    """
    value = np.reshape(value, (int(len(value)/len(energy)), len(energy))).astype(np.float32)
    value = np.nan_to_num(value)
    #print("Value post reshape: ", value)
    energy = np.array(energy, dtype=float)
    fig, ax = plt.subplots(1,1)
    c = ax.pcolormesh(energy, energy, value, cmap='rainbow', vmin=vmin, vmax=vmax, shading='nearest')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r'$\hbar\omega_{\tau}$, [eV]', fontsize=20)
    ax.set_ylabel(r'$\hbar\omega_{t}$, [eV]', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    #plt.show()
    plt.savefig(outname, bbox_inches='tight', dpi=400)
    plt.close()

