import matplotlib
import matplotlib.pyplot as plt
from plot_utils import *


def main():
    sn = 'SN2007on'

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    osc = pd.read_csv('ref/osc.csv', index_col='Name')

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    xmin = xmax = 0
    handles = labels = []
    flux_exp = lum_exp = 0

    # Plot GALEX data
    bands = ['FUV', 'NUV']
    colors = ['m', 'b']
    marker_styles = ['D', 'o']

    for band, color, marker in zip(bands, colors, marker_styles):
        # Import light curve file, if it exists
        try:
            lc, flag_count = get_lc_data(sn, band, sn_info)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            continue

        # Skip if no useful data points found or not enough background info
        if len(lc.index) == 0 or len(lc[lc['t_delta'] < 0].index) == 0:
            continue

        # Get host background levels & errors
        bg, bg_err, sys_err = get_background(lc)
        bg_lum, bg_err_lum, sys_err_lum = absolute_luminosity(sn, np.array([bg, bg_err, sys_err]), sn_info)

        # Add systematics to lc error bars
        lc['luminosity_err'] = np.sqrt(lc['luminosity_err'] ** 2 + sys_err_lum ** 2)
        lc['flux_bgsub_err'] = np.sqrt(lc['flux_bgsub_err'] ** 2 + sys_err ** 2)
        lc['luminosity_hostsub'] = lc['luminosity'] - bg_lum
        lc['flux_hostsub'] = lc['flux_bgsub'] - bg

        # Get max exponent
        flux_exp = np.max(flux_exp, int(np.log10(np.max(lc['flux_hostsub']))))

        # Set xaxis limits
        xmax = max((xmax, np.max(lc['t_delta'])))
        xmin = min((xmin, np.min(lc['t_delta'])))

        # Plot background average of epochs before discovery
        # ax.axhline(y=bg_err_lum, alpha=0.8, color=color, label=band+' host 2σ background')
        ax.axhline(bg, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1, label=band+' host background')
        ax.fill_between(x=[-4000, 4000], y1=bg - 2 * bg_err, y2=bg + 2 * bg_err, 
                color=color, alpha=0.2, label=band+' host 2σ')

        # Plot fluxes
        # lc_data = lc[lc['luminosity'] > 0]
        markers, caps, bars = ax.errorbar(lc['t_delta'], lc['flux_bgsub'], 
                yerr=lc['flux_bgsub_err'], linestyle='none', marker=marker, ms=4,
                elinewidth=1, c=color, label=band+' flux'
        )
        [bar.set_alpha(0.8) for bar in bars]

        handles, labels = ax.get_legend_handles_labels()

    # Configure plot
    if len(labels) > 0 and len(lc.index) > 0:
        ax.set_xlabel('Time since discovery [days]')
        ax.set_xlim((xmin - 50, xmax + 50))
        flux_exp = ax.get_yaxis().get_offset_text()
        print(flux_exp)
        ax.set_ylabel('Flux [erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$]')
        plt.legend(handles, labels)
        fig.suptitle(sn)
        plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png'))
        short_range = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < 1000)]
        if len(short_range.index) > 0:
            xlim = (short_range['t_delta'].iloc[0]-20, short_range['t_delta'].iloc[-1]+20)
            ax.set_xlim(xlim)
            plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png'))
    plt.show()



if __name__ == '__main__':
    main()