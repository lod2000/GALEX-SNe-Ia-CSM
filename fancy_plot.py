#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from lc_utils import *
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(description='Generate publication-quality plots of light curves for individual SNe.')
    parser.add_argument('sne', metavar='SN', type=str, nargs='+', help='supernova name')
    parser.add_argument('-b', '--band', type=str, choices=['FUV', 'NUV', 'both'], 
            default='both', help='GALEX bands to plot')
    parser.add_argument('-s', '--sigma', type=float, default=3, 
            help='number of sigma to plot as host background')
    parser.add_argument('-m', '--max', type=float, default=4000,
            help='maximum numer of days after discovery to plot')
    parser.add_argument('-i', '--info', type=Path, default=Path('ref/sn_info.csv'),
            help='path to sn info csv file', metavar='file.csv')
    parser.add_argument('-S', '--show', action='store_true',
            help='show each plot before saving')
    parser.add_argument('-o', '--output', type=Path, default=Path('figs/'),
            help='output directory', metavar='dir')
    parser.add_argument('-e', '--external', action='store_true',
            help='also plot external light curves')
    args = parser.parse_args()

    sn_info = pd.read_csv(args.info, index_col='name')

    for sn in args.sne:
        try:
            plot(sn, sn_info, args)
        except FileNotFoundError:
            print('%s is missing at least one LC file! Skipping for now.' % sn)
        # except ValueError:
        #     print('%s is missing a Swift data entry, or something else went wrong.' % sn)


def plot(sn, sn_info, args):
    """
    Plots light curve(s) for given SN
    Inputs:
        args: parser arguments
    """

    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    nearest_epoch = sn_info.loc[sn, 'delta_t_next']
    last_epoch = sn_info.loc[sn, 'delta_t_last']

    bands = ['FUV', 'NUV'] if args.band == 'both' else [args.band]
    colors = {'FUV': 'm', 'NUV': 'b', 'UVW1': 'maroon', 'UVM2': 'orange', 'UVW2': 'g'}

    data = [full_import(sn, band, sn_info) for band in bands]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_tight_layout(True)

    # Get largest flux exponent
    fluxes = np.concatenate([lc['flux_bgsub'].to_numpy() for lc in data])
    flux_exp = int(np.log10(np.max(fluxes))) - 1
    yscale = 1 / (10**flux_exp)

    # Plot points after discovery
    for lc, band in zip(data, bands):
        color = colors[band]
        after = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < args.max)]

        # Systematics
        bg, bg_err, sys_err = get_background(lc, band, 'flux_bgsub')

        # Plot background average of epochs before discovery
        bg_alpha = 0.2
        plt.axhline(bg * yscale, 0, 1, color=color, alpha=0.5, linestyle='--', 
                linewidth=1)
        plt.axhspan(ymin=(bg - args.sigma * bg_err) * yscale, 
                ymax=(bg + args.sigma * bg_err) * yscale, 
                color=color, alpha=bg_alpha)

        # Plot fluxes
        ax.errorbar(after['t_delta'], after['flux_bgsub'] * yscale, 
                yerr=after['flux_bgsub_err_total'] * yscale, linestyle='none', 
                marker='o', ms=5,
                elinewidth=1, c=color, label=band
        )

    # Indicate points before discovery
    xmin = ax.get_xlim()[0]
    for lc, band in zip(data, bands):
        before = lc[lc['t_delta'] <= DT_MIN]
        before_t = [xmin] * len(before.index)
        ax.scatter(before_t, before['flux_bgsub'] * yscale, marker='<', s=15,
                c=colors[band], label='%s host (%s)' % (band, len(before.index)))

    # Plot external light curves (e.g. Swift)
    if args.external:
        lc = import_swift_lc(sn, sn_info)
        filters = ['UVW1', 'UVM2', 'UVW2']
        for f in filters:
            data = lc[lc['band'] == f]
            ax.errorbar(data['t_delta'], data['flux'] * yscale, linestyle='none',
                    yerr=data['flux_err'] * yscale, marker='D', ms=4, label=f,
                    elinewidth=1, color=colors[f])

    # Configure plot
    ax.set_xlabel('Time since discovery [days]')
    ax.set_ylabel('Flux [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$ cm$^{-2}$]' % flux_exp)
    ylim_flux = np.array(ax.get_ylim()) * 10**flux_exp

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Greyscale line, and patch for host background flux
    bg_line = mlines.Line2D([], [], color='k', linestyle='--', alpha=0.5,
            label='host mean', linewidth=1)
    bg_patch = mpatches.Patch(color='k', alpha=bg_alpha, label='host %sσ' % args.sigma)
    # Add handles from fluxes
    plt.legend(handles=[bg_line, bg_patch] + handles, ncol=3, 
            loc='upper right', handletextpad=0.5, handlelength=1.2)

    # Twin axis with absolute luminosity
    luminosity_ax = ax.twinx()
    ylim_luminosity = absolute_luminosity(ylim_flux, sn_info.loc[sn, 'h_dist'])
    luminosity_exp = int(np.log10(max(ylim_luminosity)))
    luminosity_ax.set_ylim(ylim_luminosity / (10**luminosity_exp))
    luminosity_ax.set_ylabel('Luminosity [$10^{%s}$ erg s$^{-1}$ Å$^{-1}$]' % luminosity_exp, 
            rotation=270, labelpad=24)

    plt.savefig(args.output / Path('%s.png' % sn))
    if args.show: plt.show()


def full_import(sn, band, sn_info):
    """
    Imports the light curve for a specified supernova and band, adds luminosity
    and days since discovery from SN info file, and incorporates background
    and systematic errors
    """

    lc = import_lc(sn, band)
    lc = improve_lc(lc, sn, sn_info)
    lc = add_systematics(lc, band, 'all')
    return lc


if __name__ == '__main__':
    main()