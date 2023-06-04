import numpy as np
from .transform import Mass_Spin


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


# ===============================================
# Plotting
# ===============================================

def plot_corner(samples, show_titles=True, **kwargs):
    import corner
    plot_range = []
    for key in samples.keys():
        a = np.percentile(samples[key], 0.5)
        b = np.percentile(samples[key], 99.5)
        plot_range.append((a, b))
    return corner.corner(samples,
                         show_titles=show_titles,
                         range=plot_range,
                         **kwargs)


def plot_grid(res, posterior_samples=True):
    from matplotlib import pyplot as plt
    if posterior_samples:
        try:
            m1 = res.posterior_samples["mass_1"]
            m2 = res.posterior_samples["mass_2"]
            x = Mass_Spin.from_m1m2(m1,
                                    m2,
                                    grid_coordinates=res.grid_coordinates)
            plt.scatter(x.x1, x.x2, alpha=0.1, c="k", s=1)
        except AttributeError:
            pass

    vmin = res.marg_log_likelihood.min()
    vmax = res.marg_log_likelihood.max()
    for gl in res.iteration:
        plt.scatter(res.x1[res.iteration == gl],
                    res.x2[res.iteration == gl],
                    marker="+",
                    s=60/(gl+1),
                    vmin=vmin,
                    vmax=vmax,
                    c=res.marg_log_likelihood[res.iteration == gl])
    plt.colorbar(label=r"$\ln\mathcal{L}_{\mathrm{marg}}$")
    plt.xlabel(res.grid_coordinates[0])
    plt.ylabel(res.grid_coordinates[1])
