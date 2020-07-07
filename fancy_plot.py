import matplotlib
import matplotlib.pyplot as plt
from lc_utils import *


def main():
    sn = 'SN2007on'
    xmax = None
    band = 'NUV'

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    osc = pd.read_csv('ref/osc.csv', index_col='Name')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # Plot GALEX data
    data = []
    bands = []

    if band == 'FUV' or band == 'both':
        data.append(import_lc(sn, 'FUV'))
        bands.append('FUV')
    if band == 'NUV' or band == 'both':
        data.append(import_lc(sn, 'NUV'))
        bands.append('NUV')

    # Get largest flux exponent
    fluxes = np.concatenate([lc['flux_bgsub'].to_numpy() for lc in data])
    flux_exp = int(np.log10(np.max(fluxes))) - 1
    yscale = 1 / (10**flux_exp)

    for lc, band in zip(data, bands):
        # Plot styling
        if band == 'FUV':
            color = 'm'
            fs = 'none'
        else:
            color = 'b'
            fs = 'full'

        # Add time relative to discovery date
        disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
        lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd

        # Add systematics
        bg, bg_err, sys_err = get_background(lc)
        lc = add_systematics(lc, bg, bg_err, sys_err)

        # Plot all points before discovery on a compressed scale
        before = lc[lc['t_delta'] <= DT_MIN]
        before_dt = np.linspace(DT_MIN - 50, DT_MIN, len(before.index))
        ax.errorbar(before_dt, before['flux_bgsub'] * yscale, 
                yerr=before['flux_bgsub_err_total'] * yscale, marker='<', ms=3,
                elinewidth=1, c=color, linestyle='none', fillstyle=fs)

        lc = lc[lc['t_delta'] > DT_MIN]

        # Plot background average of epochs before discovery
        ax.axhline(bg * yscale, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1, label=band+' host background')
        ax.axhspan(ymin=(bg - 2 * bg_err) * yscale, 
                ymax=(bg + 2 * bg_err) * yscale, 
                color=color, alpha=0.2, label=band+' host 2σ')

        # Plot fluxes
        ax.errorbar(lc['t_delta'], lc['flux_bgsub'] * yscale, 
                yerr=lc['flux_bgsub_err'] * yscale, linestyle='none', 
                marker='o', ms=5, fillstyle=fs,
                elinewidth=1, c=color, label=band+' flux'
        )

    # Configure plot
    ax.set_xlabel('Time since discovery [days]')
    # ax.set_xlim((DT_MIN - 50, xmax))
    ax.set_ylabel('Flux [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$]' % flux_exp)
    plt.legend()
    # plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png'))
    # short_range = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < 1000)]
    # if len(short_range.index) > 0:
    #     xlim = (short_range['t_delta'].iloc[0]-20, short_range['t_delta'].iloc[-1]+20)
    #     ax.set_xlim(xlim)
        # plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png'))
    plt.show()


if __name__ == '__main__':
    main()