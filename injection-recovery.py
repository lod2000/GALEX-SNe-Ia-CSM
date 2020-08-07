from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt

from corner import corner

from utils import *
from CSMmodel import CSMmodel

def main(args, tstart=300, twidth=300, decay_rate=0.3, scale=1):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

    # Initialize arrays
    params = []
    nondet = []

    # List of all possible combinations of SNe and model parameters
    sne = sn_info.index.to_numpy()
    lists = [sne, tstart, twidth, decay_rate, scale]
    comb = list(itertools.product(*lists))
    iterations = min((args.iter, len(comb)))

    # Randomly sample SNe and parameters
    sample = [comb.pop(np.random.randint(0, len(comb))) for i in range(iterations)]

    with Pool() as pool:
        func = partial(count_recovered, bins=args.bins, sn_info=sn_info, sigma=args.sigma)
        for sample_params, counts in tqdm(pool.imap(func, sample, chunksize=10), total=iterations):
            params.append(sample_params)
            nondet.append(counts)

    # Combine data
    midx = pd.MultiIndex.from_tuples(params, names=('tstart', 'twidth', 'decay_rate', 'scale'))
    # print(midx.levels)
    df = pd.DataFrame(np.vstack(nondet), index=midx, columns=args.bins[:-1])
    df.sort_index(inplace=True)
    # print(df)
    # [df.reset_index(level=name, drop=True, inplace=True) for name in midx.names if len(midx.get_level_values(name)) == 1]
    # print(df)
    sums = df.groupby(df.index).sum()
    print(sums)


def count_recovered(sample_params, bins, sn_info, sigma):
    """Count recovered detections from injection-recovery for given model 
    parameters.
    """

    # Initialize nondetection counts
    nondet = np.full(len(bins)-1, 0)

    # Unpack sample parameters
    sn, tstart, twidth, decay_rate, scale = sample_params

    # Choose bands with light curve data
    bands = [b for b in ['FUV', 'NUV'] if (LC_DIR / sn2fname(sn, b, suffix='.csv')).is_file()]

    for band in bands:
        # Get nondetection epochs from injection-recovery
        try:
            t = inject_recover(sn, band, sn_info, sigma, tstart, twidth, decay_rate, scale)
        except (KeyError, pd.errors.EmptyDataError):
            # In case of empty light curve file
            continue

        # Split recovered epochs by bin and record recovered detections per bin
        if np.nan in t:
            print(t)
        n_det = np.array(
                [len(t[(t > bins[i]) & (t < bins[i+1])]) for i in range(len(bins)-1)]
        )

        # Convert counts to true/false per bin
        mask = n_det > 0
        nondet += mask.astype(int)

    # Return parameters and nondetections
    return (tstart, twidth, decay_rate, scale), nondet


def inject_recover(sn, band, sn_info, sigma, tstart, twidth, decay_rate, scale):
    """Perform injection and recovery for given SN and model parameters."""

    z = sn_info.loc[sn, 'z']
    lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
    lc = inject_model(lc, band, z, tstart, twidth, decay_rate, scale)
    recovered = recover_model(lc, sigma)
    return recovered['t_delta_rest']


def inject_model(lc, band, z, tstart, twidth, decay_rate, scale):
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


def recover_model(lc, sigma=3):
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
    parser.add_argument('--bins', '-b', type=int, nargs='+', default=[0, 100, 500, 2500],
            help='Epoch bin times for statistics, including upper bound')
    parser.add_argument('--decay', '-D', nargs='+', default=[0.3], type=float, 
            help='Fractional decay rate per 100 days')
    parser.add_argument('--scale', '-s', type=float, nargs='+', default=[1],
            help='Multiplicative scale factor for CSM model')
    parser.add_argument('--sigma', '-S', type=float, default=3, 
            help='Detection significance level')
    parser.add_argument('--detections', '-d', nargs='+', default=[0, 0, 0], type=int,
            help='Number of detections in each bin; must pass one fewer argument than number of bins')
    args = parser.parse_args()

    # Define parameter space
    param_space = { 'tstart': np.arange(args.tmin, args.tmax, args.tstep),
                    'twidth': np.arange(args.wmin, args.wmax, args.wstep),
                    'decay_rate': args.decay,
                    'scale': args.scale
                    }

    main(args, **param_space)