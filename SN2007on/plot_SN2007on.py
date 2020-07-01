import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time

external_lc = pd.read_csv('SN2007on_phot.dat')
swift = external_lc[external_lc['telescope'] == 'Swift']
fuv = pd.read_csv('SN2007on-FUV.csv')
nuv = pd.read_csv('SN2007on-NUV.csv')

swift_times = swift['time']
swift_mjd = Time(swift_times, format='mjd')
swift.insert(2, 'time_iso', swift_mjd.iso)
fuv['t_mean_iso'] = Time(fuv['t_mean'], format='gps').iso
nuv['t_mean_iso'] = Time(nuv['t_mean'], format='gps').iso
fuv['t_mean_mjd'] = Time(fuv['t_mean'], format='gps').mjd
nuv['t_mean_mjd'] = Time(nuv['t_mean'], format='gps').mjd

# Split op Swift data into bands
uvw1 = swift[swift['band'] == 'UVW1']
uvw2 = swift[swift['band'] == 'UVW2']
uvm2 = swift[swift['band'] == 'UVM2']

xshift = -54408

# Conversions from Swift Vega to AB mags from Breeveld et al. 2011
uvw1.insert(4, 'ab_mag', uvw1['magnitude'] + 1.51)
uvw1.insert(5, 'ab_mag_err', np.sqrt(uvw1['e_magnitude']**2 + 0.03**2))
uvm2.insert(4, 'ab_mag', uvm2['magnitude'] + 1.69)
uvm2.insert(5, 'ab_mag_err', np.sqrt(uvm2['e_magnitude']**2 + 0.03**2))
uvw2.insert(4, 'ab_mag', uvw2['magnitude'] + 1.73)
uvw2.insert(5, 'ab_mag_err', np.sqrt(uvw2['e_magnitude']**2 + 0.03**2))

# Plot
x_list = [uvw1['time']+xshift, uvm2['time']+xshift, uvw2['time']+xshift, 
        fuv['t_mean_mjd']+xshift, nuv['t_mean_mjd']+xshift]
y_list = [uvw1['ab_mag'], uvm2['ab_mag'], uvw2['ab_mag'], fuv['mag_bgsub'], nuv['mag_bgsub']]
yerr_list = [uvw1['ab_mag_err'], uvm2['ab_mag_err'], uvw2['ab_mag_err'], 
        [fuv['mag_bgsub_err_2'], fuv['mag_bgsub_err_1']], [nuv['mag_bgsub_err_2'], 
        nuv['mag_bgsub_err_1']]]
formats = ['cp','gp','yp','mD','bo']
labels = ['Swift W1', 'Swift M2', 'Swift W2', 'GALEX FUV', 'GALEX NUV']

for x, y, yerr, f, label in zip(x_list, y_list, yerr_list, formats, labels):
    markers, caps, bars = plt.errorbar(x, y, yerr=yerr, fmt=f, label=label,
            linestyle='none', capsize=0, elinewidth=1, ms=4)
    [bar.set_alpha(0.8) for bar in bars]

plt.xlim((0, 72))
plt.xlabel('MJD' + str(xshift))
plt.ylim((25, 13))
plt.ylabel('AB apparent magnitude')
plt.legend()

plt.savefig('SN2007on.png')
plt.show()