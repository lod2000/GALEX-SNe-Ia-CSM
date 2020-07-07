import matplotlib
import matplotlib.pyplot as plt
from plot_utils import *


def main():
    sn = 'ESSENCEn278'
    xmax = None

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    osc = pd.read_csv('ref/osc.csv', index_col='Name')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # Plot GALEX data
    bands = ['FUV', 'NUV']
    colors = ['m', 'b']

    FUVdata = import_lc(sn, 'FUV')
    NUVdata = import_lc(sn, 'NUV')

    # Get largest flux exponent
    fluxes = np.concatenate([FUVdata['flux_bgsub'].to_numpy(), NUVdata['flux_bgsub'].to_numpy()])
    flux_exp = int(np.log10(np.max(fluxes))) - 1
    yscale = 1 / (10**flux_exp)

    for data, band, color in zip([FUVdata, NUVdata], bands, colors):
        # Add time relative to discovery date
        disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
        data['t_delta'] = data['t_mean_mjd'] - disc_date.mjd

        # Add systematics
        bg, bg_err, sys_err = get_background(data)
        data = add_systematics(data, bg, bg_err, sys_err)

        # Plot all points before discovery on a compressed scale
        before = data[data['t_delta'] <= DT_MIN]
        before_dt = np.linspace(DT_MIN - 50, DT_MIN, len(before.index))
        ax.errorbar(before_dt, before['flux_bgsub'] * yscale, 
                yerr=before['flux_bgsub_err_total'] * yscale, marker='<', ms=4,
                elinewidth=1, c=color, linestyle='none')

        data = data[data['t_delta'] > DT_MIN]

        # Plot background average of epochs before discovery
        ax.axhline(bg * yscale, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1, label=band+' host background')
        ax.axhspan(ymin=(bg - 2 * bg_err) * yscale, 
                ymax=(bg + 2 * bg_err) * yscale, 
                color=color, alpha=0.2, label=band+' host 2σ')

        # Plot fluxes
        ax.errorbar(data['t_delta'], data['flux_bgsub'] * yscale, 
                yerr=data['flux_bgsub_err'] * yscale, linestyle='none', 
                marker='o', ms=6,
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


def galex_data(sn, band, sn_info):
    # Import light curve file, if it exists
    try:
        lc = import_lc(sn, band)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        return None


    # Skip if no useful data points found or not enough background info
    if len(lc.index) == 0 or len(lc[lc['t_delta'] < 0].index) == 0:
        return None

    return lc, bg, bg_err


if __name__ == '__main__':
    main()