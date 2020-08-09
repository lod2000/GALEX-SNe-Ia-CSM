from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from corner import corner

from utils import *
from CSMmodel import CSMmodel

# Default values
BINS = [0, 100, 500, 2500]
DECAY_RATE = 0.3
DETECTIONS = [0, 0, 0]
SCALE = 1
SIGMA = 3

def main(iterations, tstart, twidth, decay_rate=DECAY_RATE, scale=SCALE, 
        bins=BINS, det=DETECTIONS, sigma=SIGMA, overwrite=False):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    output_file = Path('out/recovery.csv')

    # List of all possible combinations of SNe and model parameters
    sne = [Supernova(sn, sn_info) for sn in sn_info.index.to_list()]
    lists = [sne, tstart, twidth]
    comb = list(itertools.product(*lists))
    iterations = min((iterations, len(comb)))

    # Randomly sample SNe and parameters
    sample = [comb.pop(np.random.randint(0, len(comb))) for i in range(iterations)]

    if output_file.is_file() or overwrite:
        sums = pd.read_csv(output_file, index_col=['tstart', 'twidth'])
    else:
        sums = sum_recovered(sample, decay_rate, scale, bins, sigma)
        sums.to_csv(Path('out/recovery.csv'))

    plot_recovered(sums)


def plot_recovered(sums):
    """Plot a heatmap of recoveries in each epoch bin."""
    
    fig, axs = plt.subplots(1, len(sums.columns), sharey=True)

    for ax, col in zip(axs, sums.columns):
        arr = sums[col].unstack()
        xlevels = sums.index.levels[1]
        dx = xlevels[1] - xlevels[0]
        ylevels = sums.index.levels[0]
        dy = ylevels[1] - ylevels[0]
        im = ax.imshow(arr, vmin=0, vmax=sums.max().max(), origin='lower',
                extent=[xlevels[0], xlevels[-1]+dx, ylevels[0], ylevels[-1]+dy])
        ax.set_title(col)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(axis='both', labelcolor='none', which='both', top=False, 
            bottom=False, left=False, right=False)
    plt.xlabel(sums.index.names[1])
    plt.ylabel(sums.index.names[0])

    fig.subplots_adjust(right=0.8, wspace=0.05)
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im, cax=cax, label='no. recovered', use_gridspec=True)

    # plt.tight_layout()
    plt.savefig(Path('out/recovered.png'), dpi=300)
    plt.close()


def sum_recovered(sample, decay_rate=DECAY_RATE, scale=SCALE, bins=BINS, sigma=SIGMA):
    """Run injection-recovery on all parameters of given sample."""

    # Initialize arrays
    params = []
    recovered = []

    with Pool() as pool:
        func = partial(count_recovered, decay_rate=decay_rate, scale=scale,
                bins=bins, sigma=sigma)
        imap = pool.imap(func, sample, chunksize=10)
        for sample_params, counts in tqdm(imap, total=len(sample)):
            params.append(sample_params)
            recovered.append(counts)

    # Combine data
    midx = pd.MultiIndex.from_tuples(params, names=('tstart', 'twidth'))
    col_names = ['%s-%s' % (bins[i], bins[i+1]) for i in range(len(bins) - 1)]
    df = pd.DataFrame(np.vstack(recovered), index=midx, columns=col_names)
    df.sort_index(inplace=True)

    # Sum data
    sums = df.groupby(df.index).sum()
    sums_midx = pd.MultiIndex.from_tuples(sums.index, names=('tstart', 'twidth'))
    sums.set_index(sums_midx, drop=True, inplace=True)

    return sums


def count_recovered(sample_params, decay_rate=DECAY_RATE, scale=SCALE, 
        bins=BINS, sigma=SIGMA):
    """Count recovered detections from injection-recovery for given model 
    parameters.
    """

    # Initialize recovery counts
    recovered = np.full(len(bins)-1, 0)

    # Unpack sample parameters
    sn, tstart, twidth = sample_params

    # Choose bands with light curve data
    bands = [b for b in ['FUV', 'NUV'] if (LC_DIR / sn2fname(sn.name, b, suffix='.csv')).is_file()]

    for band in bands:
        # Get nondetection epochs from injection-recovery
        try:
            t = inject_recover(sn, band, tstart, twidth, decay_rate=decay_rate, 
                    scale=scale, sigma=sigma)
        except (KeyError, pd.errors.EmptyDataError):
            # In case of empty light curve file
            continue

        # Split recovered epochs by bin and record recovered detections per bin
        n_det = np.array(
                [len(t[(t > bins[i]) & (t < bins[i+1])]) for i in range(len(bins)-1)]
        )

        # Convert counts to true/false per bin
        mask = n_det > 0
        recovered += mask.astype(int)

    # Return parameters and nondetections
    return (tstart, twidth), recovered


def inject_recover(sn, band, tstart, twidth, decay_rate=DECAY_RATE, scale=SCALE, 
        sigma=SIGMA):
    """Perform injection and recovery for given SN and model parameters."""

    lc, bg, bg_err, sys_err = full_import_2(sn, band)
    lc = inject_model(lc, band, sn.z, tstart, twidth, decay_rate=decay_rate, 
            scale=scale)
    recovered = recover_model(lc, sigma=sigma)
    return recovered['t_delta_rest']


def inject_model(lc, band, z, tstart, twidth, decay_rate=DECAY_RATE, scale=SCALE):
    """
    Inject CSM model into GALEX data and return the resulting light curve
    Inputs:
        lc: light curve DataFrame
        band: GALEX filter
        tstart: days after discovery that ejecta impacts CSM 
        width: length of light curve plateau in days
        scale: luminosity scale factor
    """

    model = CSMmodel(tstart, twidth, decay_rate, scale=scale)
    # Calculate luminosity at observation epochs
    injection = model(lc['t_delta_rest'], z)[band]
    # Inject CSM curve
    lc['luminosity_injected'] = lc['luminosity_hostsub'] + injection
    return lc


def recover_model(lc, sigma=SIGMA):
    """
    Recover detections from CSM-injected data which otherwise wouldn't have
    been detected; recovered data must also be after discovery
    """

    lc['sigma_injected'] = lc['luminosity_injected'] / lc['luminosity_hostsub_err']
    recovered = lc[(lc['sigma_injected'] > sigma) & (lc['sigma'] < sigma)]
    recovered = recovered[recovered['t_delta_rest'] > 0]
    return recovered


def corner_plot(sums, bin):
    """Display a corner plot of model parameter samples."""

    params = np.vstack(sums.index.to_numpy())
    counts = np.array([sums[bin].to_numpy()]).T
    data = np.hstack((params[:,0:2], counts))
    
    fig = corner(data, labels=['tstart', 'twidth', 'counts'])
    plt.show()


class Supernova:
    def __init__(self, name, sn_info=[], fname=Path('ref/sn_info.csv')):
        """Initialize Supernova by importing reference file."""

        if len(sn_info) == 0:
            sn_info = pd.read_csv(Path(fname), index_col='name')

        self.name = name
        self.data = sn_info.loc[name].to_dict()

        self.z = self.data['z']
        self.z_err = self.data['z_err']
        self.dist = self.data['pref_dist']
        self.dist_err = self.data['pref_dist_err']
        self.disc_date = Time(self.data['disc_date'], format='iso')
        self.a_v = self.data['a_v']

    def __call__(self, key=None):
        """Return value associated with key."""

        if key == None:
            return self.data
        return self.data[key]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('iter', type=int, help='Iterations')
    # tstart parameters
    parser.add_argument('--tmin', '-t0', default=0, type=int,
            help='Minimum CSM model interaction start time, in days post-discovery')
    parser.add_argument('--tmax', '-t1', default=1000, type=int,
            help='Maximum CSM model interaction start time, in days post-discovery')
    parser.add_argument('--tstep', '-dt', default=100, type=int,
            help='Start time interation step in days')
    # twidth parameters
    parser.add_argument('--wmin', '-w0', default=100, type=int,
            help='Minimum CSM plateau width in days')
    parser.add_argument('--wmax', '-w1', default=600, type=int,
            help='Maximum CSM plateau width in days')
    parser.add_argument('--wstep', '-dw', default=100, type=int,
            help='Plateau width interation step in days')
    # other parameters
    parser.add_argument('--decay', '-D', default=DECAY_RATE, 
            type=float, help='Fractional decay rate per 100 days')
    parser.add_argument('--scale', '-s', type=float, default=SCALE,
            help='Multiplicative scale factor for CSM model')
    parser.add_argument('--bins', '-b', type=int, nargs='+', default=BINS,
            help='Epoch bin times for statistics, including upper bound')
    parser.add_argument('--detections', '-d', nargs='+', default=DETECTIONS, 
            type=int, help='Number of detections in each bin; must pass one '\
            + 'fewer argument than number of bins')
    parser.add_argument('--sigma', '-S', type=float, default=SIGMA, 
            help='Detection significance level')
    parser.add_argument('--overwrite', '-o', action='store_true',
            help='Overwrite sums')
    args = parser.parse_args()

    # Define parameter space
    tstart = np.arange(args.tmin, args.tmax, args.tstep)
    twidth = np.arange(args.wmin, args.wmax, args.wstep)

    # Keyword arguments
    kwargs = {'decay_rate': args.decay, 'scale': args.scale, 'bins': args.bins, 
            'det': args.detections, 'sigma': args.sigma, 'overwrite': args.overwrite}

    main(args.iter, tstart, twidth, **kwargs)