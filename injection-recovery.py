from tqdm import tqdm
import itertools
from multiprocessing import Pool
from functools import partial

import warnings

from utils import *
from CSMmodel import CSMmodel

DECAY_RATE = 0.3
SCALE_FACTORS = [0.1, 0.5, 1, 2]
WIDTH = 200 # plateau width (days)
SIGMA = 3 # significance of data above which it counts as a detection
DETECTIONS = [0, 0, 0]

def main(iterations, tstart_max, dtstart, decay_rate, bins):

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

    # Initialize arrays
    params = []
    nondet = []

    sne = sn_info.index.to_numpy()
    start_times = np.arange(0, tstart_max, dtstart) # days post-discovery
    widths = np.arange(100, 600, 100) # days
    scales = np.array([1])

    # List of all possible combinations of SNe and model parameters
    lists = [sne, start_times, widths, scales]
    comb = list(itertools.product(*lists))

    # Sample SNe and parameters
    sample = [comb.pop(np.random.randint(0, len(comb))) for i in range(iterations)]

    with Pool() as pool:
        func = partial(count_recovered, sn_info=sn_info, bins=bins, decay_rate=decay_rate)
        for sample, counts in tqdm(pool.imap(func, sample, chunksize=10), total=iterations):
            params.append(sample)
            nondet.append(counts)

    # for i in tqdm(range(iterations)):
    #     sample, counts = count_recovered(comb, sn_info, bins, decay_rate)
    #     params.append(sample)
    #     nondet.append(counts)

    # Combine data
    midx = pd.MultiIndex.from_tuples(params, names=('tstart', 'twidth', 'scale'))
    df = pd.DataFrame(np.vstack(nondet), index=midx, columns=bins[:-1])
    df.sort_values(by=['tstart', 'twidth'], axis=0, inplace=True)
    sums = df.groupby(df.index).sum()
    print(sums)


def count_recovered(sample_params, sn_info, bins, decay_rate):
    """Count recovered detections from injection-recovery for given model 
    parameters.
    """

    # Initialize nondetection counts
    nondet = np.full(len(bins)-1, 0)

    # Unpack sample parameters
    sn, tstart, twidth, scale = sample_params

    # Choose bands with light curve data
    bands = [b for b in ['FUV', 'NUV'] if (LC_DIR / sn2fname(sn, b, suffix='.csv')).is_file()]

    for band in bands:
        # Get nondetection epochs from injection-recovery
        try:
            t = inject_recover(sn, band, sn_info, tstart, twidth, decay_rate, scale)
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
    return (tstart, twidth, scale), nondet


def inject_recover(sn, band, sn_info, tstart, twidth, decay_rate, scale):
    """Perform injection and recovery for given SN and model parameters."""

    z = sn_info.loc[sn, 'z']
    lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
    lc = inject_model(lc, band, z, tstart, twidth, decay_rate, scale)
    recovered = recover_model(lc)
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


def recover_model(lc):
    """
    Recover detections from CSM-injected data which otherwise wouldn't have
    been detected; recovered data must also be after discovery
    """

    lc['sigma_injected'] = lc['luminosity_injected'] / lc['luminosity_hostsub_err']
    recovered = lc[(lc['sigma_injected'] > SIGMA) & (lc['sigma'] < SIGMA)]
    recovered = recovered[recovered['t_delta_rest'] > 0]
    return recovered


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmax', '-m', default=1000, type=int,
            help='Maximum CSM model interaction start time, in days post-discovery')
    parser.add_argument('--dtstart', '-t', default=100, type=int,
            help='Start time interation step in days')
    parser.add_argument('--bins', '-b', type=int, nargs='+', default=[0, 100, 500, 2500],
            help='Epoch bin times for statistics, including upper bound')
    parser.add_argument('--iter', '-i', type=int, default=1000, help='Iterations')
    parser.add_argument('--decay-rate', '-D', default=0.3, type=float, 
            help='fractional decay rate per 100 days')
    args = parser.parse_args()

    main(args.iter, args.tmax, args.dtstart, args.decay_rate, args.bins)