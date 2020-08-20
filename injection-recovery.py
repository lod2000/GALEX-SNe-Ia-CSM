from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from corner import corner

from utils import *
from CSMmodel import CSMmodel

# Default values
BINS = [0, 100, 500, 2500]
DECAY_RATE = 0.3
DETECTIONS = [0, 0, 0]
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
SIGMA = 3
WIDTH = 250 # days, from PTF11kx
SCALE = 0

# def main(iterations, tstart, scale, decay_rate=DECAY_RATE, scale=SCALE, 
#         bins=BINS, det=DETECTIONS, sigma=SIGMA, overwrite=False):
def main():

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    output_file = Path('out/recovery.csv')

    supernovae = ['SN2007on', 'SN2010ai']
    # supernovae = sn_info.index.to_list()

    recovered_times = run_ir(100, supernovae, 0, 1000, 0.5, 2)
    rate_hist = get_recovery_rate(recovered_times, 50, 0.1)
    # print(rate_hist)
    plot_recovery_rate(rate_hist)

    # recovered_times = inject_recover(sn, 'FUV', 0, 1)
    # print(recovered_times)

    # # List of all possible combinations of SNe and model parameters
    # sne = [Supernova(sn, sn_info) for sn in sn_info.index.to_list()]
    # lists = [sne, tstart, twidth]
    # comb = list(itertools.product(*lists))
    # iterations = min((iterations, len(comb)))

    # # Randomly sample SNe and parameters
    # sample = [comb.pop(np.random.randint(0, len(comb))) for i in range(iterations)]

    # if output_file.is_file() or overwrite:
    #     sums = pd.read_csv(output_file, index_col=['tstart', 'twidth'])
    # else:
    #     sums = sum_recovered(sample, decay_rate, scale, bins, sigma)
    #     sums.to_csv(Path('out/recovery.csv'))

    # plot_recovered(sums)


# def plot_recovered(sums):
#     """Plot a heatmap of recoveries in each epoch bin."""
    
#     fig, axs = plt.subplots(1, len(sums.columns), sharey=True)

#     for ax, col in zip(axs, sums.columns):
#         arr = sums[col].unstack()
#         xlevels = sums.index.levels[1]
#         dx = xlevels[1] - xlevels[0]
#         ylevels = sums.index.levels[0]
#         dy = ylevels[1] - ylevels[0]
#         im = ax.imshow(arr, vmin=0, vmax=sums.max().max(), origin='lower',
#                 extent=[xlevels[0], xlevels[-1]+dx, ylevels[0], ylevels[-1]+dy])
#         ax.set_title(col)

#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(axis='both', labelcolor='none', which='both', top=False, 
#             bottom=False, left=False, right=False)
#     plt.xlabel(sums.index.names[1])
#     plt.ylabel(sums.index.names[0])

#     fig.subplots_adjust(right=0.8, wspace=0.05)
#     cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     plt.colorbar(im, cax=cax, label='no. recovered', use_gridspec=True)

#     # plt.tight_layout()
#     plt.savefig(Path('out/recovered.png'), dpi=300)
#     plt.close()


# def sum_recovered(sample, decay_rate=DECAY_RATE, scale=SCALE, bins=BINS, sigma=SIGMA):
#     """Run injection-recovery on all parameters of given sample."""

#     # Initialize arrays
#     params = []
#     recovered = []

#     with Pool() as pool:
#         func = partial(count_recovered, decay_rate=decay_rate, scale=scale,
#                 bins=bins, sigma=sigma)
#         imap = pool.imap(func, sample, chunksize=10)
#         for sample_params, counts in tqdm(imap, total=len(sample)):
#             params.append(sample_params)
#             recovered.append(counts)

#     # Combine data
#     midx = pd.MultiIndex.from_tuples(params, names=('tstart', 'twidth'))
#     col_names = ['%s-%s' % (bins[i], bins[i+1]) for i in range(len(bins) - 1)]
#     df = pd.DataFrame(np.vstack(recovered), index=midx, columns=col_names)
#     df.sort_index(inplace=True)

#     # Sum data
#     sums = df.groupby(df.index).sum()
#     sums_midx = pd.MultiIndex.from_tuples(sums.index, names=('tstart', 'twidth'))
#     sums.set_index(sums_midx, drop=True, inplace=True)

#     return sums


# def count_recovered(sample_params, decay_rate=DECAY_RATE, scale=SCALE, 
#         bins=BINS, sigma=SIGMA):
#     """Count recovered detections from injection-recovery for given model 
#     parameters.
#     """

#     # Initialize recovery counts
#     recovered = np.full(len(bins)-1, 0)

#     # Unpack sample parameters
#     sn, tstart, twidth = sample_params

#     # Choose bands with light curve data
#     bands = [b for b in ['FUV', 'NUV'] if (LC_DIR / sn2fname(sn.name, b, suffix='.csv')).is_file()]

#     for band in bands:
#         # Get nondetection epochs from injection-recovery
#         try:
#             t = inject_recover(sn, band, tstart, twidth, decay_rate=decay_rate, 
#                     scale=scale, sigma=sigma)
#         except (KeyError, pd.errors.EmptyDataError):
#             # In case of empty light curve file
#             continue

#         # Split recovered epochs by bin and record recovered detections per bin
#         n_det = np.array(
#                 [len(t[(t > bins[i]) & (t < bins[i+1])]) for i in range(len(bins)-1)]
#         )

#         # Convert counts to true/false per bin
#         mask = n_det > 0
#         recovered += mask.astype(int)

#     # Return parameters and nondetections
#     return (tstart, twidth), recovered


def plot_recovery_rate(rate_hist):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor."""

    # Flip y-axis
    rate_hist.sort_index(ascending=True, inplace=True)
    # Calculate data range
    bin_width = rate_hist.columns[1] - rate_hist.columns[0]
    bin_height = rate_hist.index[1] - rate_hist.index[0]
    extent = (rate_hist.columns[0], rate_hist.columns[-1]+bin_width, 
            rate_hist.index[0], rate_hist.index[-1]+bin_height)
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(rate_hist, aspect='auto', origin='lower', extent=extent)
    ax.xaxis.set_minor_locator(MultipleLocator(bin_width))
    ax.yaxis.set_minor_locator(MultipleLocator(bin_height))
    ax.set_xlabel('Time since discovery [days]')
    ax.set_ylabel('Scale factor')
    plt.colorbar(im)
    fig.tight_layout()
    plt.show()


def get_recovery_rate(recovered_times, bin_width, bin_height):
    """Generate 2D histogram of recovery rate by time and scale factor
    Inputs:
        recovered_times: list of dicts
        bin_width: bin width in heatmap plot, in days
        bin_height: bin height in heatmap plot, in scale factor fraction
    Output:
        rate_hist: 2D histogram of recovery rates, DataFrame
    """
    
    print('\nBinning recovery rates...')
    # Make lists of recovered points and all points
    recovered = []
    total = []
    for rec_dict in tqdm(recovered_times):
        recovered += [[time, rec_dict['scale']] for time in rec_dict['recovered']]
        total += [[time, rec_dict['scale']] for time in rec_dict['all']]

    recovered = np.array(recovered)
    total = np.array(total)

    # 2D histogram
    x_edges = np.arange(RECOV_MIN, np.max(total[:,0]), bin_width)
    y_edges = np.arange(np.min(total[:,1]), np.max(total[:,1]), bin_height)
    rec_hist = np.histogram2d(recovered[:,0], recovered[:,1], [x_edges, y_edges])[0]
    total_hist = np.histogram2d(total[:,0], total[:,1], [x_edges, y_edges])[0]
    rate_hist = rec_hist / total_hist
    # Transpose and convert to DataFrame with time increasing along the rows
    # and scale height increasing down the columns. Column and index labels
    # are the lower bound of each bin
    rate_hist = pd.DataFrame(rate_hist.T, index=y_edges[:-1], columns=x_edges[:-1])

    return rate_hist


def run_ir(iterations, supernovae, tstart_min, tstart_max, scale_min, scale_max):
    """Run injection recovery with random parameters for a list of supernovae.
    Inputs:
        iterations: number of times to sample parameter space
        supernovae: list of SN names
        tstart_min: minimum CSM model start time
        tstart_max: maximum CSM model start time
        scale_min: minimum CSM model scale factor
        scale_max: maximum CSM model scale factor
    Outputs:
        recovered_times: list of dicts
    """

    # List of supernovae and bands to perform injection-recovery
    supernovae = sorted(list(supernovae) * 2)
    bands = ['FUV', 'NUV'] * len(supernovae)
    recovered_times = []

    # Iterate over supernovae, bands
    for i, (sn_name, band) in enumerate(zip(supernovae, bands)):
        # Ignore if light curve file doesn't exist
        lc_file = LC_DIR / sn2fname(sn_name, band)
        if not lc_file.is_file():
            continue

        print('\n%s - %s [%s/%s]' % (sn_name, band, i+1, len(supernovae)))
        sn = Supernova(sn_name)
        # Run injection-recovery on many randomly sampled parameters
        sample_times = sample_params(iterations, sn, band, tstart_min, 
                tstart_max, scale_min, scale_max)
        # Append resulting recovered times
        recovered_times += sample_times

    return recovered_times


def sample_params(iterations, sn, band, tstart_min, tstart_max, scale_min, scale_max):
    """Run injection recovery on a single SN for a given number of iterations.
    Inputs:
        iterations: int
        sn: Supernova object
        band: 'FUV' or 'NUV'
        tstart_min: minimum CSM model start time
        tstart_max: maximum CSM model start time
        scale_min: minimum CSM model scale factor
        scale_max: maximum CSM model scale factor
    Outputs:
        sample_times: list of dicts
    """

    # Randomly sample start times (ints) and scale factors (floats)
    tstarts = np.random.randint(tstart_min, tstart_max, size=iterations)
    scales = (scale_max - scale_min) * np.random.rand(iterations) + scale_min
    params = np.array(list(zip(tstarts, scales)))

    # Import light curve for SN
    lc = LightCurve(sn, band)
    all_times = lc.data[lc.data['t_delta_rest'] >= RECOV_MIN]['t_delta_rest'].to_list()
    sample_times = []

    # Run injection-recovery in parallel for each sampled CSM parameter
    with Pool() as pool:
        func = partial(inject_recover, sn=sn, lc=lc)
        imap = pool.imap(func, params, chunksize=10)
        for times in tqdm(imap, total=params.shape[0]):
            sample_times.append(times)

    # List of recovered times and associated parameters
    sample_times = [
            {   'sn': sn.name,
                'band': band,
                'tstart': params[i,0], 
                'scale': params[i,1], 
                'recovered': sample_times[i],
                'all': all_times} 
            for i in range(iterations)]

    return sample_times


def inject_recover(params, sn, lc):
    """Perform injection and recovery for given SN and model parameters.
    Inputs:
        params: [tstart, scale]
        sn: Supernova object
        lc: LightCurve object
    Output:
        list of times of recovered data
    """

    tstart, scale = params
    injected = inject_model(sn, lc, tstart, scale)
    recovered = recover_model(injected)
    # Return days post-discoverey with recovered detections
    return recovered['t_delta_rest'].to_list()


def inject_model(sn, lc, tstart, scale):
    """
    Inject CSM model into GALEX data and return the resulting light curve.
    Inputs:
        sn: Supernova object
        lc: LightCurve object
        tstart: days after discovery that ejecta impacts CSM
        scale: luminosity scale factor
    Output:
        lc: LightCurve object with injected light curve
    """

    data = lc.data.copy()
    model = CSMmodel(tstart, WIDTH, DECAY_RATE, scale=scale)
    # Calculate luminosity at observation epochs
    injection = model(data['t_delta_rest'], sn.z)[lc.band]
    # Inject CSM curve
    data['luminosity_injected'] = data['luminosity_hostsub'] + injection
    return data


def recover_model(data):
    """
    Recover detections from CSM-injected data which otherwise wouldn't have
    been detected.
    """

    # Calculate significance of each point
    data['sigma_injected'] = data['luminosity_injected'] / data['luminosity_hostsub_err']
    # Recover new detections
    recovered = data[(data['sigma_injected'] >= SIGMA) & (data['sigma'] < SIGMA)]
    # Limit to points some time after discovery (default 50 days)
    recovered = recovered[recovered['t_delta_rest'] >= RECOV_MIN]
    return recovered


# def corner_plot(sums, bin):
#     """Display a corner plot of model parameter samples."""

#     params = np.vstack(sums.index.to_numpy())
#     counts = np.array([sums[bin].to_numpy()]).T
#     data = np.hstack((params[:,0:2], counts))
    
#     fig = corner(data, labels=['tstart', 'twidth', 'counts'])
#     plt.show()


class LightCurve:
    def __init__(self, sn, band):
        self.band = band
        self.data, self.bg, self.bg_err, self.sys_err = full_import_2(sn, band)

    def __call__(self):
        return self.data

    @classmethod
    def from_fname(self, fname):
        sn_name, self.band = fname2sn(fname)
        sn = Supernova(sn_name)
        return LightCurve(sn, self.band)


class Supernova:
    def __init__(self, name, sn_info=[], fname='ref/sn_info.csv'):
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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('iter', type=int, help='Iterations')
    # # tstart parameters
    # parser.add_argument('--tmin', '-t0', default=0, type=int,
    #         help='Minimum CSM model interaction start time, in days post-discovery')
    # parser.add_argument('--tmax', '-t1', default=1000, type=int,
    #         help='Maximum CSM model interaction start time, in days post-discovery')
    # # parser.add_argument('--tstep', '-dt', default=100, type=int,
    # #         help='Start time interation step in days')
    # # twidth parameters
    # parser.add_argument('--wmin', '-w0', default=100, type=int,
    #         help='Minimum CSM plateau width in days')
    # parser.add_argument('--wmax', '-w1', default=600, type=int,
    #         help='Maximum CSM plateau width in days')
    # parser.add_argument('--wstep', '-dw', default=100, type=int,
    #         help='Plateau width interation step in days')
    # # scale parameters
    # parser.add_argument('--smin', '-w0', default=100, type=float,
    #         help='Minimum CSM plateau width in days')
    # parser.add_argument('--smax', '-w1', default=600, type=float,
    #         help='Maximum CSM plateau width in days')
    # # other parameters
    # parser.add_argument('--decay', '-D', default=DECAY_RATE, 
    #         type=float, help='Fractional decay rate per 100 days')
    # # parser.add_argument('--scale', '-s', type=float, default=SCALE,
    # #         help='Multiplicative scale factor for CSM model')
    # parser.add_argument('--bins', '-b', type=int, nargs='+', default=BINS,
    #         help='Epoch bin times for statistics, including upper bound')
    # parser.add_argument('--detections', '-d', nargs='+', default=DETECTIONS, 
    #         type=int, help='Number of detections in each bin; must pass one '\
    #         + 'fewer argument than number of bins')
    # parser.add_argument('--sigma', '-S', type=float, default=SIGMA, 
    #         help='Detection significance level')
    # parser.add_argument('--overwrite', '-o', action='store_true',
    #         help='Overwrite sums')
    # args = parser.parse_args()

    # # Define parameter space
    # tstart = np.arange(args.tmin, args.tmax, args.tstep)
    # twidth = np.arange(args.wmin, args.wmax, args.wstep)

    # # Keyword arguments
    # kwargs = {'decay_rate': args.decay, 'scale': args.scale, 'bins': args.bins, 
    #         'det': args.detections, 'sigma': args.sigma, 'overwrite': args.overwrite}

    # main(args.iter, tstart, twidth, **kwargs)
    main()