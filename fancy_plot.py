import matplotlib
import matplotlib.pyplot as plt
from lc_utils import *


def main():
    sn = 'ESSENCEn278'
    dt_max = 10000
    band = 'both'

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')

    fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1,10], 'wspace': 0.1})
    before_ax = ax[0]
    data_ax = ax[1]
    fig.set_tight_layout(True)

    # Plot GALEX data
    data = []
    bands = []
    before_t0 = 0

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
        else:
            color = 'b'

        # Add time relative to discovery date
        lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd

        # Add systematics
        bg, bg_err, sys_err = get_background(lc)
        lc = add_systematics(lc, bg, bg_err, sys_err)

        before = lc[lc['t_delta'] <= DT_MIN]
        lc = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < dt_max)]

        # Plot background average of epochs before discovery
        plt.axhline(bg * yscale, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1, label=band+' host background')
        plt.axhspan(ymin=(bg - 2 * bg_err) * yscale, 
                ymax=(bg + 2 * bg_err) * yscale, 
                color=color, alpha=0.2, label=band+' host 2σ')

        # Plot fluxes
        data_ax.errorbar(lc['t_delta'], lc['flux_bgsub'] * yscale, 
                yerr=lc['flux_bgsub_err_total'] * yscale, linestyle='none', 
                marker='o', ms=5,
                elinewidth=1, c=color, label=band+' flux'
        )

        # Plot all points before discovery on a compressed scale
        # before_dt = np.linspace(lc['t_delta'].iloc[0] - 100, 
        #         lc['t_delta'].iloc[0] - 50, len(before.index))
        before_dt = np.arange(0, len(before.index)) + before_t0
        before_t0 += len(before.index) 
        # before_dt = np.linspace(before_t0, before_t0+1, num=len(before.index), endpoint=False)
        before_ax.errorbar(before_dt, before['flux_bgsub'] * yscale, 
                yerr=before['flux_bgsub_err_total'] * yscale, marker='<', ms=4,
                elinewidth=1, c=color, linestyle='none')

    # Configure plot
    data_ax.set_xlabel('Time since discovery [days]')
    data_ax.spines['left'].set_visible(False)
    data_ax.tick_params(axis='y', which='both', left=False, right=False)
    # data_ax.yaxis.set_ticks_position('right')

    before_ax.set_ylabel('Flux [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$]' % flux_exp)
    before_ax.spines['right'].set_visible(False)
    before_ax.tick_params(axis='y', which='both', left=True, right=False, labelright=False)
    before_ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    ylim_flux = np.array(data_ax.get_ylim()) * 10**flux_exp
    plt.legend()

    # Twin axis with absolute luminosity
    luminosity_ax = data_ax.twinx()
    # ylim_luminosity = absolute_luminosity(ylim_flux, sn_info.loc[sn, 'h_dist'])
    # luminosity_exp = int(np.log10(max(ylim_luminosity)))
    # luminosity_ax.set_ylim(ylim_luminosity / (10**luminosity_exp))
    # luminosity_ax.set_ylabel('Luminosity [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$]' % luminosity_exp, 
    #         rotation=270, labelpad=24)
    luminosity_ax.spines['left'].set_visible(False)
    luminosity_ax.tick_params(axis='y', which='both', left=False, right=True, labelleft=False)
    # luminosity_ax.set_yticks([])

    # plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png'))
    # short_range = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < 1000)]
    # if len(short_range.index) > 0:
    #     xlim = (short_range['t_delta'].iloc[0]-20, short_range['t_delta'].iloc[-1]+20)
    #     ax.set_xlim(xlim)
        # plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png'))
    plt.show()


if __name__ == '__main__':
    main()