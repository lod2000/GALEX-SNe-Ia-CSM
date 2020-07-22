#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from utils import *
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
    parser.add_argument('-l', '--log', action='store_true',
            help='plot light curves in log(flux)')
    parser.add_argument('-p', '--pad', action='store_true',
            help='extra padding for legend at the top-right')
    parser.add_argument('-r', '--restframe', action='store_true',
            help='plot rest frame time, corrected for z')
    parser.add_argument('--notitle', action='store_true', help='omit axis title')
    args = parser.parse_args()

    sn_info = pd.read_csv(args.info, index_col='name')

    for sn in args.sne:
        try:
            plot(sn, sn_info, args)
        except FileNotFoundError:
            print('%s is missing at least one LC file! Skipping for now.' % sn)
            continue


def plot(sn, sn_info, args):
    """
    Plots light curve(s) for given SN
    Inputs:
        args: parser arguments
    """

    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')

    bands = ['FUV', 'NUV'] if args.band == 'both' else [args.band]

    data = []
    exclude = []
    for band in bands:
        try:
            lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
            data.append(lc)
        except FileNotFoundError:
            print('Could not find %s light curve file for %s; skipping for now.' % (band, sn))
            exclude.append(band)

    for e in exclude: bands.remove(e)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_tight_layout(True)

    # Get largest flux exponent
    fluxes = np.concatenate([lc['flux_bgsub'].to_numpy() for lc in data])
    if args.log:
        flux_exp = 0
        yscale = 1
    else:
        flux_exp = int(np.log10(np.max(fluxes))) - 1
        yscale = 1 / (10**flux_exp)

    # Plot external light curves (e.g. Swift)
    if args.external:
        try:
            ax = plot_swift(ax, sn, sn_info, yscale, args)
        except (ValueError, FileNotFoundError):
            print('%s is missing a Swift data entry!' % sn)
        try:
            ax = plot_panstarrs(ax, sn, sn_info, yscale, args)
        except (ValueError, FileNotFoundError):
            print('%s is missing a Pan-STARRS data entry!' % sn)


    # Plot points after discovery
    for lc, band in zip(data, bands):
        color = COLORS[band]
        after = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < args.max)]

        # Systematics
        bg, bg_err, sys_err = get_background(lc, band)

        # Plot background average of epochs before discovery
        bg_alpha = 0.2
        plt.axhline(bg * yscale, 0, 1, color=color, alpha=0.7, linestyle='--', 
                linewidth=1)
        plt.axhspan(ymin=(bg - args.sigma * bg_err) * yscale, 
                ymax=(bg + args.sigma * bg_err) * yscale, 
                color=color, alpha=bg_alpha)

        # Plot fluxes
        if args.restframe:
            dt = after['t_delta_rest']
            xlabel = 'Rest frame time since discovery [days]'
        else:
            dt = after['t_delta']
            xlabel = 'Time since discovery [days]'
        ax.errorbar(dt, after['flux_bgsub'] * yscale, 
                yerr=after['flux_bgsub_err_total'] * yscale, linestyle='none', 
                marker='o', ms=5, alpha=1,
                elinewidth=1, c=color, label=band
        )

    # Indicate points before discovery
    xmin = ax.get_xlim()[0]
    for lc, band in zip(data, bands):
        before = lc[lc['t_delta'] <= DT_MIN]
        before_t = [xmin] * len(before.index)
        ax.scatter(before_t, before['flux_bgsub'] * yscale, marker='<', s=15,
                c=COLORS[band], label='%s host (%s)' % (band, len(before.index)))

    # Configure plot
    ax.set_xlabel(xlabel)
    if args.log:
        flux_exp_text = ''
        ax.set_yscale('log')
        ylim_flux = np.array(ax.get_ylim()) * 10**flux_exp
        if args.pad:
            ylim_flux[1] *= 3
            ax.set_ylim(ylim_flux)
    else:
        flux_exp_text = '$10^{%s}$ ' % flux_exp
        ax.ticklabel_format(useOffset=False)
        ylim_flux = np.array(ax.get_ylim()) * 10**flux_exp
    ax.set_ylabel('Flux Density [%serg s$^{-1}$ Å$^{-1}$ cm$^{-2}$]' % flux_exp_text)
    if not args.notitle:
        ax.set_title(sn)
    if args.pad:
        xlim = np.array(ax.get_xlim())
        xlim[1] += 0.2 * (xlim[1] - xlim[0])
        ax.set_xlim(xlim)
        legend_loc = 'upper right'
    else:
        legend_loc = 'best'

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Greyscale line, and patch for host background flux
    bg_line = mlines.Line2D([], [], color='k', linestyle='--', alpha=0.7,
            label='host mean', linewidth=1)
    bg_patch = mpatches.Patch(color='k', alpha=bg_alpha, label='host %sσ' % args.sigma)
    # Add handles from fluxes
    ncol = 2 if len(handles) < 4 else 3
    plt.legend(handles=[bg_line, bg_patch] + handles, ncol=ncol, 
            loc=legend_loc, handletextpad=0.5, handlelength=1.2)

    # Twin axis with absolute luminosity
    luminosity_ax = ax.twinx()
    ylim_luminosity = absolute_luminosity(ylim_flux, sn_info.loc[sn, 'pref_dist'])
    if args.log:
        luminosity_ax.set_yscale('log')
        luminosity_ax.set_ylim(ylim_luminosity)
        luminosity_exp_text = ''
    else:
        luminosity_exp = int(np.log10(max(ylim_luminosity)))
        luminosity_ax.set_ylim(ylim_luminosity / (10**luminosity_exp))
        luminosity_exp_text = '$10^{%s}$ ' % luminosity_exp
    
    luminosity_ax.set_ylabel('Luminosity [%serg s$^{-1}$ Å$^{-1}$]' % luminosity_exp_text, 
            rotation=270, labelpad=24)

    plt.savefig(args.output / Path('%s.png' % sn), dpi=300)
    if args.show: plt.show()


def plot_swift(ax, sn, sn_info, yscale, args):
    lc = import_swift_lc(sn, sn_info)
    lc = lc[lc['t_delta'] <= args.max]
    bands = ['UVW1', 'UVM2', 'UVW2']
    for band in bands:
        data = lc[lc['band'] == band]
        if args.restframe:
            dt = data['t_delta_rest']
        else:
            dt = data['t_delta']
        ax.errorbar(data['t_delta'], data['flux'] * yscale, linestyle='none',
                yerr=data['flux_err'] * yscale, marker='D', ms=4, label=band,
                elinewidth=1, markeredgecolor=COLORS[band], markerfacecolor='white',
                ecolor=COLORS[band])
    return ax


def plot_panstarrs(ax, sn, sn_info, yscale, args):
    lc = import_panstarrs(sn, sn_info)
    lc = lc[lc['t_delta'] <= args.max]
    filter_ids = np.arange(1, 6)
    bands = ['g', 'r', 'i', 'z', 'y']
    for i in filter_ids:
        data = lc[lc['filterID'] == i]
        ax.errorbar(data['t_delta'], data['apFlux_cgs'] * yscale, 
                yerr=data['apFluxErr_cgs'] * yscale, marker='*', ms=4, 
                label=bands[i-1], elinewidth=1, color=COLORS[bands[i-1]],
                linestyle='none')
    return ax


def plot_background(ax):
    return ax


if __name__ == '__main__':
    main()