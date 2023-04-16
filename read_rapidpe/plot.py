def pretty_plt(plt, usetex=True):
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
