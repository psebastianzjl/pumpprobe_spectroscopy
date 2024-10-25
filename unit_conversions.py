
def fs2aut(fs):
    """
     Function that converts time from femto-seconds to atomic units.
     Parameters
     ----------
     fs : float
     Time in femto-seconds that is to be converted to atomic units of time

     Returns
     -------
     Time in atomic units
     """
    aut = fs * 41.341373336
    return aut

def aut2fs(aut):
    """
     Function that converts time from atomic units of time to femto-seconds.
     Parameters
     ----------
     aut : float
     Time in atomic units that is to be converted to femto-seconds

     Returns
     -------
     Time in femto-seconds
     """
    fs = aut / 41.341373336
    return fs

def eV2Ha(eV):
    """
    Function that converts energy from electron Volt (eV) to atomic units (hartree).
    Parameters
    ----------
    eV : float
    Energy in eV

    Returns
    -------
    Energy in atomic units (hartree)
    """
    Ha = eV / 27.21138602
    return Ha

def Ha2eV(Ha):
    """
    Function that converts energy from atomic units (hartree) to electron Volts (eV).
    Parameters
    ----------
    Ha : float
    Energy in atomic units (hartree)

    Returns
    -------
    Energy in atomic units (hartree)
    """
    eV = Ha * 27.21138602
    return eV

def ang2bohr(ang):
    """
    Function that converts Angstrom to bohr radius
    Parameters
    ----------
    ang : float
    Distance in Angstrom that is to be converted to bohr radius

    Returns
    -------
    Distance in bohr radius
    """
    bohr = ang * 1.889726125
    return bohr

def bohr2ang(bohr):
    """
    Function that converts bohr radius to Angstrom
    Parameters
    ----------
    bohr : float
    Distance in bohr radius to be converted to Angstrom

    Returns
    -------
    Distance in Angstrom
    """
    ang = bohr / 1.889726125
    return ang

def dal2el(dal):
    """
    Function that converts mass from Dalton to electron mass
    Parameters
    ----------
    dal : float
    Mass in Dalton to be converted into electron mass

    Returns
    -------
    Mass in electron mass
    """
    el = dal * 1822.88849
    return el

def el2dal(el):
    """
    Function that converts mass from electron mass to Dalton
    Parameters
    ----------
    el : float
    Mass in electron mass to be converted into Dalton

    Returns
    -------
    Mass in Dalton
    """
    dal = el / 1822.88849
    return dal