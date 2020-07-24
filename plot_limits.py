import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import astropy.units as u
from utils import *

overwrite = False
plot_systematics = True


def main():
    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')
    conf_det = pd.read_csv(Path('out/confirmed_detections.csv'))
    det_sne = list(zip(conf_det['Name'], conf_det['Band']))

    if overwrite or not Path('out/nondetections.csv').is_file():
        print('Separating detections from nondetections...')

        # Combine detections
        detections = []
        for sn, band in det_sne:
            lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
            lc = add_uniform_columns(lc, [sn, band, bg, bg_err, sys_err], 
                    ['name', 'band', 'host', 'host_err', 'sys_err'])
            lc['sigma'] = lc['flux_hostsub'] / lc['flux_hostsub_err']
            detections.append(lc)
        detections = pd.concat(detections, ignore_index=True)
        output_csv(detections, Path('out/detections.csv'), index=False)

        # Combine nondetections
        bands = ['FUV', 'NUV'] * len(sn_info.index)
        sne = np.array([[sn] * 2 for sn in sn_info.index]).flatten()
        nondet_sne = zip(sne, bands)
        # Remove detections
        nondet_sne = [loc for loc in nondet_sne if not loc in det_sne]
        with mp.Pool() as pool:
            nondetections = list(tqdm(
                pool.imap(partial(get_nondetection, sn_info=sn_info), 
                    nondet_sne, chunksize=10), 
                total=len(nondet_sne)
            ))
        nondetections = [lc for lc in nondetections if len(lc.index) > 0]
        nondetections = pd.concat(nondetections, ignore_index=True)
        output_csv(nondetections, Path('out/nondetections.csv'), index=False)
    else:
        detections = pd.read_csv(Path('out/detections.csv'))
        nondetections = pd.read_csv(Path('out/nondetections.csv'))

    if plot_systematics:
        print('Plotting systematics...')
        # Look for systematics in observations
        all_detections = nondetections.append(detections)
        plot_observation_systematics(all_detections, sn_info)
        # Look for systematics in SN sample
        plot_sample_systematics(sn_info)

    print('Plotting detections & limits...')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ebar_alpha = 0.8
    limit_alpha = 0.6 # downward triangle: 3-sigma limit
    detection_cutoff = 3 # sigma
    limit_sigma = 3
    nondet_alpha = 0.1

    markers = ['o', 's', 'p', 'd', 'X', 'P', '*']

    # Plot detections
    for i, (sn, band) in enumerate(det_sne):
        lc = detections[(detections['name'] == sn) & (detections['band'] == band)]
        lc_det = lc[lc['sigma'] > detection_cutoff]
        ebar = ax.errorbar(lc_det['t_delta_rest'], lc_det['luminosity_hostsub'],
                yerr=lc_det['luminosity_hostsub_err'], linestyle='none', 
                label='%s (%s)' % (sn, band), marker=markers[i], ms=6, elinewidth=1)
        [bar.set_alpha(ebar_alpha) for bar in ebar[2]]
        lc_non = lc[lc['sigma'] <= detection_cutoff]
        ax.scatter(lc_non['t_delta_rest'], limit_sigma * lc_non['luminosity_hostsub_err'],
                marker='v', s=36, color=ebar[0].get_color(), alpha=limit_alpha)

    # Plot nondetections
    for band in ['FUV', 'NUV']:
        lc = nondetections[nondetections['band'] == band]
        ax.scatter(lc['t_delta_rest'], limit_sigma * lc['luminosity_hostsub_err'],
                marker='v', s=25, color=COLORS[band], alpha=nondet_alpha)
        # ax.scatter(lc['t_delta'], 3 * lc['luminosity_hostsub_err'],
        #         marker='v', s=25, color=COLORS[band], alpha=limit_alpha)

    ax.set_xlabel('Rest frame time since discovery [days]')
    # ax.set_xlabel('Observed time since discovery [days]')
    ax.set_xlim((-50, None))
    ax.set_ylabel('Luminosity [erg s$^{-1}$ Å$^{-1}$]')
    ax.set_yscale('log')

    plt.legend()
    plt.savefig('out/limits.png', dpe=300)
    plt.show()


def add_uniform_columns(df, values, col_names):
    new_cols = np.full([len(df.index), len(values)], values)
    new_cols = pd.DataFrame(new_cols, columns=col_names, index=df.index)
    df = df.join(new_cols)
    return df


def get_nondetection(loc, sn_info):
    sn, band = loc
    try:
        lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
        lc = add_uniform_columns(lc, [sn, band, bg, bg_err, sys_err], 
                ['name', 'band', 'host', 'host_err', 'sys_err'])
        return lc
    except (FileNotFoundError, KeyError):
        return pd.DataFrame([])


def plot_observation_systematics(all_detections, sn_info):
    """
    Plots the distribution of GALEX observations over time
    """

    fig, ax = plt.subplots(2,2, figsize=(13, 8))
    fig.set_tight_layout(True)

    # Distribution of GALEX observations of targets (GPS time)
    gps_sub = 740063000
    ax[0, 0].hist(all_detections['t_mean'] - gps_sub, bins=100)
    ax[0, 0].set_xlabel('Observation date [GPS-%s]' % gps_sub)
    ax[0, 0].set_ylabel('# of obs.')

    # Distribution of GALEX observations (MJD)
    ax[0, 1].hist(all_detections['t_mean_mjd'], bins=100)
    ax[0, 1].set_xlabel('Observation date [MJD]')
    ax[0, 1].set_ylabel('# of obs.')

    # Distribution of SN discovery dates among targets
    ax[1, 0].hist(Time(sn_info['disc_date'].to_list(), format='iso').mjd, bins=100)
    ax[1, 0].set_xlabel('Discovery date [MJD]')
    ax[1, 0].set_ylabel('# of SNe')

    # Distribution of GALEX observations of targets (time since discovery)
    ax[1, 1].hist(all_detections['t_delta'], bins=100)
    ax[1, 1].set_xlabel('Observed time since discovery [days]')
    ax[1, 1].set_ylabel('# of obs.')

    plt.savefig(Path('out/obs_dist.png'), dpi=300)
    plt.close()

    all_detections.sort_values('t_mean_mjd', inplace=True)
    gaps = all_detections['t_mean_mjd'].diff()
    print('Longest gap between consecutive observations: %s days' % np.max(gaps))
    all_detections.sort_values('t_delta', inplace=True)
    gaps = all_detections['t_delta'].diff()
    print('Longest gap in phase between consecutive points: %s days' % np.max(gaps))
    gaps = all_detections[all_detections['t_delta'] > 0]['t_delta'].diff()
    print('Longest phase gap post-discovery: %s days' % np.max(gaps))



def plot_sample_systematics(sn_info):
    """
    Plots the distribution of sky positions and discovery dates for SNe in the
    final target sample, as well as from the initial OSC sample
    """

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    fig.set_tight_layout(True)

    osc = pd.read_csv(Path('ref/osc.csv'), index_col='Name')

    ra = [Angle(ang, unit=u.hourangle).hour for ang in osc['R.A.'].to_list()]
    ax[0, 0].hist(ra, bins=100)
    ax[0, 0].set_ylabel('# of SNe in OSC')

    dec = [Angle(ang, unit=u.deg).deg for ang in osc['Dec.'].to_list()]
    ax[0, 1].hist(dec, bins=100)

    disc_date = Time(osc['Disc. Date'].to_list())
    disc_date = [d.mjd for d in disc_date if d.mjd > 40000]
    ax[0, 2].hist(disc_date, bins=100)

    ra = [Angle(ang).hour for ang in sn_info['galex_ra'].to_list()]
    ax[1, 0].hist(ra, bins=100)
    ax[1, 0].set_xlabel('R.A. [hours]')
    ax[1, 0].set_ylabel('# of SNe in sample')

    dec = [Angle(ang).deg for ang in sn_info['galex_dec'].to_list()]
    ax[1, 1].hist(dec, bins=100)
    ax[1, 1].set_xlabel('Dec. [deg]')

    # Distribution of SN discovery dates among targets
    ax[1, 2].hist(Time(sn_info['disc_date'].to_list(), format='iso').mjd, bins=100)
    ax[1, 2].set_xlabel('Discovery date [MJD]')

    plt.savefig(Path('out/target_coord_dist.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()