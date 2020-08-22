from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from corner import corner

from utils import *
from CSMmodel import CSMmodel

# Default values
# BINS = [0, 100, 500, 2500]
DECAY_RATE = 0.3
RECOV_MIN = 50 # minimum number of days after discovery to count as recovery
SIGMA = 3
WIDTH = 250 # days, from PTF11kx

def main(iterations, overwrite=False, tstart_max=1000, scale_min=0.5, 
            scale_max=2., t_max=1500, bin_width=50, bin_height=0.1):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    output_file = Path('out/recovery_%s.csv' % iterations)

    # supernovae = ['SN2007on', 'SN2010ai', 'SDSS-II SN 779', 'Hawk', 'HST04Sas']
    supernovae = sn_info.index.to_list()

    if overwrite or not output_file.is_file():
        recovered_times = run_ir(iterations, supernovae, 0, tstart_max, 
                scale_min, scale_max, bin_width, bin_height, t_max, output_file)
        count_hist = count_nondetections(recovered_times, bin_width, t_max, 
                bin_height, scale_min, scale_max)
        output_csv(count_hist, output_file)
    else:
        count_hist = pd.read_csv(output_file, index_col=0)

    plot_nondetections(count_hist, show=True)


def plot_nondetections(rate_hist, show=False):
    """Plot 2D histogram of recovery rate by time since discovery and scale factor."""

    # Flip y-axis
    rate_hist.sort_index(ascending=True, inplace=True)

    # Calculate data range
    x_bins = rate_hist.columns.to_numpy(dtype=float)
    y_bins = rate_hist.index.to_numpy(dtype=float)
    bin_width = x_bins[1] - x_bins[0]
    bin_height = y_bins[1] - y_bins[0]
    extent = (x_bins[0], x_bins[-1]+bin_width, y_bins[0], y_bins[-1]+bin_height)

    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(rate_hist, aspect='auto', origin='lower', extent=extent)
    ax.xaxis.set_minor_locator(MultipleLocator(bin_width))
    ax.yaxis.set_minor_locator(MultipleLocator(bin_height))
    ax.set_xlabel('Rest frame time since discovery [days]')
    ax.set_ylabel('Scale factor')
    plt.colorbar(im, label='No. of excluded SNe Ia')
    fig.tight_layout()
    plt.savefig(Path('out/recovery.png'), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def count_nondetections(recovered_times, bin_width, x_max, bin_height, y_min, y_max):
    """Generate 2D histogram of nondetection counts by time and scale factor
    Inputs:
        recovered_times: list of dicts
        bin_width: bin width in heatmap plot, in days
        x_max: max value on x-axis
        bin_height: bin height in heatmap plot, in scale factor fraction
        y_min: minimum value on y-axis
        y_max: max value on y-axis
    Output:
        count_sum: 2D histogram of nondetection counts
    """
    
    print('\nBinning recovery rates...')
    # Make lists of recovered points and all points
    recovered = []
    total = []
    for rec_dict in tqdm(recovered_times):
        recovered += [[rec_dict['sn'], time, rec_dict['scale']] for time in rec_dict['recovered']]
        total += [[rec_dict['sn'], time, rec_dict['scale']] for time in rec_dict['all']]

    recovered = np.array(recovered)
    total = np.array(total)

    # Bin edges
    x_edges = np.arange(RECOV_MIN, x_max, bin_width)
    y_edges = np.arange(y_min, y_max, bin_height)

    # Dummy array if recovered is empty
    if recovered.shape == (0,):
        return pd.DataFrame(np.full((len(x_edges), len(y_edges)), 0), 
                index=y_edges[:-1], columns=x_edges[:-1])

    # Count nondetections per supernova
    counts = []
    for sn_name in list(dict.fromkeys(recovered[:,0])):
        # Select by sn name
        sn_rec = recovered[recovered[:,0] == sn_name][:,1:].astype(float)
        sn_tot = total[total[:,0] == sn_name][:,1:].astype(float)
        # Recovery rate histogram
        rate_hist = recovery_histogram(sn_rec[:,0], sn_rec[:,1], sn_tot[:,0], 
                sn_tot[:,1], x_edges, y_edges)
        counts.append(rate_hist)

    # Sum all histograms
    count_sum = reduce(lambda x, y: x.add(y, fill_value=0), counts)
    return count_sum


def recovery_histogram(x_recovered, y_recovered, x_total, y_total, x_edges, y_edges):
    """Generate histogram of recovery rate given x and y recovered/total values.
    Inputs:
        x_recovered: x-values for recovered data
        y_recovered: y-values for recovered data
        x_total: x-values for all data
        y_total: y-values for all data
        x_edges: bin edges for x data
        y_edges: bin edges for y data
    Output:
        rate_hist: 2D histogram of recovery rate
    """

    # 2D histograms for recovered data and total data
    recovered = np.histogram2d(x_recovered, y_recovered, [x_edges, y_edges])[0]
    total = np.histogram2d(x_total, y_total, [x_edges, y_edges])[0]

    # Calculate recovery rate
    rate_hist = recovered / total

    # Transpose and convert to DataFrame with time increasing along the rows
    # and scale height increasing down the columns. Column and index labels
    # are the lower bound of each bin
    rate_hist = pd.DataFrame(rate_hist.T, index=y_edges[:-1], columns=x_edges[:-1])
    return rate_hist


def run_ir(iterations, supernovae, tstart_min, tstart_max, scale_min, scale_max,
        bin_width, bin_height, t_max, output_file):
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
    recovered_times = []
    to_remove = []
    supernovae = sorted(list(supernovae) * 2)

    # Load progress file, if any
    progress_file = Path('out/progress_%s.npy' % iterations)
    if progress_file.is_file():
        print('\nLoading previous progress file...')
        recovered_times = list(np.load(progress_file, allow_pickle=True))
        to_remove = [(rec['sn'], rec['band']) for rec in recovered_times]

    bands = ['FUV', 'NUV'] * len(supernovae)

    # Iterate over supernovae, bands
    for i, (sn_name, band) in enumerate(zip(supernovae, bands)):

        # Skip previously run SNe
        if (sn_name, band) in to_remove:
            continue

        # Save to binary numpy file every 10 iterations (takes a long time)
        if i % 10 == 0 and i != 0:
            print('Saving progress...')
            np.save(progress_file, np.array(recovered_times))
            print('Progress saved.')

        # Plot histogram every 50 iterations
        if i % 50 == 0 and i != 0:
            rate_hist = count_nondetections(recovered_times, bin_width, t_max, 
                    bin_height, scale_min, scale_max)
            output_csv(rate_hist, output_file)
            plot_nondetections(rate_hist, show=False)

        # Ignore if light curve file doesn't exist
        lc_file = LC_DIR / sn2fname(sn_name, band)
        if not lc_file.is_file():
            print('\nNo light curve file found for %s - %s [%s/%s]' % (sn_name, band, i+1, len(supernovae)))
            recovered_times.append({
                'sn': sn_name,
                'band': band,
                'tstart': -1,
                'scale': -1,
                'recovered': [],
                'all': []
            })
            continue

        try:
            print('\n%s - %s [%s/%s]' % (sn_name, band, i+1, len(supernovae)))
            sn = Supernova(sn_name)
            # Run injection-recovery on many randomly sampled parameters
            sample_times = sample_params(iterations, sn, band, tstart_min, 
                    tstart_max, scale_min, scale_max)
            # Append resulting recovered times
            recovered_times += sample_times
        except KeyError:
            continue

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
    with Pool(processes=5) as pool:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('iter', type=int, help='Iterations')
    parser.add_argument('--overwrite', '-o', action='store_true',
            help='Overwrite recovery rate output file')
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
    args = parser.parse_args()

    # # Define parameter space
    # tstart = np.arange(args.tmin, args.tmax, args.tstep)
    # twidth = np.arange(args.wmin, args.wmax, args.wstep)

    # # Keyword arguments
    # kwargs = {'decay_rate': args.decay, 'scale': args.scale, 'bins': args.bins, 
    #         'det': args.detections, 'sigma': args.sigma, 'overwrite': args.overwrite}

    # main(args.iter, tstart, twidth, **kwargs)
    main(args.iter, overwrite=args.overwrite)