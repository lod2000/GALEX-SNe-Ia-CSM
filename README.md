# GALEX-SNe-Ia-CSM

## Big tasks

* Compile FITS file info
    * Requires: discovery dates from OSC
    * Find files with epochs before and after discovery
    * Plot histogram of # of SNe per # of epochs
    * Output csv with columns:
        * SN name
        * Discovery date
        * GALEX filter
        * GALEX sky coords
        * total epochs, epochs before, epochs after
        * Time of first and last GALEX image relative to discovery [days]
        * Time between discovery and next GALEX image [days]
* Compile SN reference info
    * Requires: list of SNe with data before + after discovery; z values and host names from OSC
    * Scrape NED for SN redshifts, distances, references, reddening, host morphology, position
        * Output to csv
    * Pull full citation info from ADS by NED refcodes
        * Output to bib file; list of catalog references
    * Generate LaTeX table with columns:
        * Redshift [from NED]
        * Hubble distance [from NED]
        * Discovery date [from OSC]
        * Time of first and last GALEX image relative to discovery
        * Time between discovery and next GALEX image
        * Host morphology [from NED + EDD]
        * Foreground reddening [from NED + EDD?]
        * Redshift-indep. distance?
        * Absolute magnitude? [from GALEX or from OSC? in UV? limit if no detection?]
        * t_max and m_max?
        * Reference(s) [from NED / ADS]
* Plot light curves

## Small tasks

* Clean up OSC csv
* Plot filter response curves for GALEX, Swift UV, HST UV

## Reference files

* fits_info: sn, band, and epochs for each FITS file provided
* sn_info: all necessary info about SN sample we want to use (before+after, z<0.5, etc.)