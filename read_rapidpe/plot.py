import numpy as np


def pretty_plt(plt, usetex=True):
    """
    import read_rapidpe.plot as rrp
    import matplotlib.pyplot as plt
    rrp.pretty_plt(plt)
    """
    # -----------------------------------------------
    # Reference:
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    # -----------------------------------------------

    # For axes (outer box)
    plt.rc('axes', labelsize='medium')  # fontsize of the x any y labels
    plt.rc('axes', linewidth=0.5)
    plt.rc('axes', labelpad=4)
    # For ticks and grid
    plt.rc('xtick.major', width=0.5, pad=7)
    plt.rc('ytick.major', width=0.5, pad=4)
    plt.rc('xtick.minor', width=0.5)
    plt.rc('ytick.minor', width=0.5)
    plt.rc('xtick', labelsize='small', direction='in', top='True')
    plt.rc('ytick', labelsize='small', direction='in', right='True')
    plt.rc('grid', linewidth=0.5, color='0.9')
    # Text and others
    if usetex:
        plt.rc('text', usetex=True)  # Latex support
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=0.5)  # Linewidth of data
    plt.rc('savefig', dpi=300)


def meshgrid_mceta(result, n=100):
    mclist = np.linspace(
        result.chirp_mass.min()+0.00001,
        result.chirp_mass.max()-0.00001,
        n)
    etalist = np.linspace(
        result.symmetric_mass_ratio.min()+0.00001,
        result.symmetric_mass_ratio.max()-0.00001,
        n)
    mc, eta = np.meshgrid(mclist, etalist)
    return mc, eta


def meshgrid_mcq(result, n=100):
    mclist = np.linspace(
        result.chirp_mass.min()+0.00001,
        result.chirp_mass.max()-0.00001,
        n)
    qlist = np.linspace(
        result.mass_ratio.min()+0.00001,
        result.mass_ratio.max()-0.00001,
        n)
    mc, q = np.meshgrid(mclist, qlist)
    return mc, q
