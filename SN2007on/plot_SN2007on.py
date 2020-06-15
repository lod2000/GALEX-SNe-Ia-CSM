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

plt.errorbar(uvw1['time']+xshift, uvw1['ab_mag'], yerr=uvw1['ab_mag_err'], 
        linestyle='none', marker='.', capsize=2, elinewidth=1, label='Swift W1')
plt.errorbar(uvm2['time']+xshift, uvm2['ab_mag'], yerr=uvm2['ab_mag_err'], 
        linestyle='none', marker='.', capsize=2, elinewidth=1, label='Swift M2')
plt.errorbar(uvw2['time']+xshift, uvw2['ab_mag'], yerr=uvw2['ab_mag_err'], 
        linestyle='none', marker='.', capsize=2, elinewidth=1, label='Swift W2')
plt.errorbar(fuv['t_mean_mjd']+xshift, fuv['mag_bgsub'], yerr=[fuv['mag_bgsub_err_2'], 
        fuv['mag_bgsub_err_1']], linestyle='none', marker='x', capsize=2, 
        elinewidth=1, label='GALEX FUV')
plt.errorbar(nuv['t_mean_mjd']+xshift, nuv['mag_bgsub'], yerr=[nuv['mag_bgsub_err_2'], 
        nuv['mag_bgsub_err_1']], linestyle='none', marker='x', capsize=2, 
        elinewidth=1, label='GALEX NUV')

plt.xlim((0, 72))
plt.xlabel('MJD ' + str(xshift))
plt.ylim((25, 13))
plt.ylabel('AB apparent magnitude')
plt.legend()

plt.show()