from tqdm import tqdm
import itertools

from utils import *
from CSMmodel import CSMmodel

DECAY_RATE = 0.3
SCALE_FACTORS = [0.1, 0.5, 1, 2]
WIDTH = 200 # plateau width (days)
SIGMA = 3 # significance of data above which it counts as a detection
DETECTIONS = [0, 0, 0]

def main(iterations, tstart_max, dtstart, bins):

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

    for i in tqdm(range(iterations)):
        sample, counts = sample_params(comb, sn_info, bins)
        params.append(sample)
        nondet.append(counts)

    midx = pd.MultiIndex.from_tuples(params, names=('tstart', 'twidth', 'scale'))
    df = pd.DataFrame(np.vstack(nondet), index=midx, columns=bins[:-1])        
    print(df)


def sample_params(comb, sn_info, bins):
    """Randomly sample parameters from list of combinations and perform 
    injection recovery. Return detections in each bin.
    """

    # Initialize nondetection counts
    nondet = np.full(len(bins)-1, 0)

    # Pick a random combination of SN and model parameters, and remove that
    # combination from future samples
    r = np.random.randint(0, len(comb))
    sn, tstart, twidth, scale = comb.pop(r)

    # Choose bands with light curve data
    bands = [b for b in ['FUV', 'NUV'] if (LC_DIR / sn2fname(sn, b, suffix='.csv')).is_file()]

    for band in bands:
        # Get nondetection epochs from injection-recovery
        try:
            t = inject_recover(sn, band, sn_info, tstart, twidth, scale=scale)
        except KeyError:
            # In case of empty light curve file
            continue

        # Split recovered epochs by bin and record recovered detections per bin
        n_det = np.array(
                [len(t[(t > bins[i]) & (t < bins[i+1])]) for i in range(len(bins)-1)]
        )

        # Convert counts to true/false per bin
        mask = n_det > 0
        nondet += mask.astype(int)

    # Return parameters and nondetections
    return (tstart, twidth, scale), nondet


def inject_recover(sn, band, sn_info, tstart, twidth, scale=1):
    """Perform injection and recovery for given SN and model parameters."""

    z = sn_info.loc[sn, 'z']
    lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
    lc = inject_model(lc, band, z, tstart, twidth, scale=scale)
    recovered = recover_model(lc)
    return recovered['t_delta_rest']


def inject_model(lc, band, z, tstart, twidth, scale=1):
    """
    Inject CSM model into GALEX data and return the resulting light curve
    Inputs:
        lc: light curve DataFrame
        band: GALEX filter
        tstart: days after discovery that ejecta impacts CSM 
        width: length of light curve plateau in days
        scale: luminosity scale factor
    """

    model = CSMmodel(tstart, twidth, DECAY_RATE, scale=scale)
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
    parser.add_argument('--dtstart', '-d', default=100, type=int,
            help='Start time interation step in days')
    parser.add_argument('--bins', '-b', type=int, nargs='+', default=[0, 100, 500, 2500],
            help='Epoch bin times for statistics, including upper bound')
    parser.add_argument('--iter', '-i', type=int, default=1000, help='Iterations')
    args = parser.parse_args()

    main(args.iter, args.tmax, args.dtstart, args.bins)