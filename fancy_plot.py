import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from lc_utils import *


def main():
    sn = 'ESSENCEn278'
    dt_max = 10000
    bands = ['FUV', 'NUV']

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    nearest_epoch = sn_info.loc[sn, 'delta_t_next']
    last_epoch = sn_info.loc[sn, 'delta_t_last']

    data = [full_import(sn, band, sn_info) for band in bands]
    colors = {'FUV': 'm', 'NUV': 'b'}

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_tight_layout(True)

    # Get largest flux exponent
    fluxes = np.concatenate([lc['flux_bgsub'].to_numpy() for lc in data])
    flux_exp = int(np.log10(np.max(fluxes))) - 1
    yscale = 1 / (10**flux_exp)

    # Plot points after discovery
    for lc, band in zip(data, bands):
        color = colors[band]
        after = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < dt_max)]

        # Systematics
        bg, bg_err, sys_err = get_background(lc, 'flux')

        # Plot background average of epochs before discovery
        bg_sigma = 3
        bg_alpha = 0.2
        plt.axhline(bg * yscale, 0, 1, color=color, alpha=0.5, linestyle='--', 
                linewidth=1)
        plt.axhspan(ymin=(bg - bg_sigma * bg_err) * yscale, 
                ymax=(bg + bg_sigma * bg_err) * yscale, 
                color=color, alpha=bg_alpha)

        # Plot fluxes
        ax.errorbar(after['t_delta'], after['flux_bgsub'] * yscale, 
                yerr=after['flux_bgsub_err_total'] * yscale, linestyle='none', 
                marker='o', ms=5,
                elinewidth=1, c=color, label=band+' flux'
        )

    # Indicate points before discovery
    xmin = ax.get_xlim()[0]
    for lc, band in zip(data, bands):
        before = lc[lc['t_delta'] <= DT_MIN]
        before_t = [xmin] * len(before.index)
        ax.scatter(before_t, before['flux_bgsub'] * yscale, marker='<', s=5,
                c=colors[band])

    # Configure plot
    ax.set_xlabel('Time since discovery [days]')
    ax.set_ylabel('Flux [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$]' % flux_exp)
    ylim_flux = np.array(ax.get_ylim()) * 10**flux_exp

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    bg_line = mlines.Line2D([], [], color='k', linestyle='--', alpha=0.5,
            label='host mean', linewidth=1)
    bg_patch = mpatches.Patch(color='k', alpha=bg_alpha, label='host %sσ' % bg_sigma)
    bg_point = mlines.Line2D([], [], color='k', linestyle='none', marker='<',
            label='host flux')
    plt.legend(handles=[bg_patch, bg_line, bg_point] + handles, ncol=5, 
            loc='lower left', mode='expand', handletextpad=0.5, handlelength=1.2,
            bbox_to_anchor=(0, 1.02, 1., 0), borderaxespad=0)
    # plt.gca().add_artist(first_legend)
    # plt.legend(loc='best')

    # Twin axis with absolute luminosity
    luminosity_ax = ax.twinx()
    ylim_luminosity = absolute_luminosity(ylim_flux, sn_info.loc[sn, 'h_dist'])
    luminosity_exp = int(np.log10(max(ylim_luminosity)))
    luminosity_ax.set_ylim(ylim_luminosity / (10**luminosity_exp))
    luminosity_ax.set_ylabel('Luminosity [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$]' % luminosity_exp, 
            rotation=270, labelpad=24)

    plt.show()


def full_import(sn, band, sn_info):

    lc = import_lc(sn, band)
    lc = improve_lc(lc, sn, sn_info)
    lc = add_systematics(lc, 'all')
    return lc


if __name__ == '__main__':
    main()