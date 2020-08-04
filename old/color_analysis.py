import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy import units as u
from utils import *

sn = 'SDSS-II SN 19778'

sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

NUVdata = full_import(sn, 'NUV', sn_info)[0]
NUVdata.set_index(NUVdata['t_delta_rest'].round(2), inplace=True)
FUVdata = full_import(sn, 'FUV', sn_info)[0]
FUVdata.set_index(FUVdata['t_delta_rest'].round(2), inplace=True)

dt = pd.concat([NUVdata['t_delta_rest'], FUVdata['t_delta']], axis=0).round(2)
dt = dt.sort_values().drop_duplicates().reset_index(drop=True)
data = pd.DataFrame([], index=dt)
data['FUVmag'] = FUVdata['mag_bgsub']
data['FUVmag_err_1'] = FUVdata['mag_bgsub_err_1']
data['FUVmag_err_2'] = FUVdata['mag_bgsub_err_2']
data['NUVmag'] = NUVdata['mag_bgsub']
data['NUVmag_err_1'] = NUVdata['mag_bgsub_err_1']
data['NUVmag_err_2'] = NUVdata['mag_bgsub_err_2']
data['color_mag'] = data['FUVmag'] - data['NUVmag']
data['color_mag_err_1'] = np.sqrt(data['FUVmag_err_1']**2 + data['NUVmag_err_1']**2)
data['color_mag_err_2'] = np.sqrt(data['FUVmag_err_2']**2 + data['NUVmag_err_2']**2)
data['FUVlum'] = FUVdata['luminosity_hostsub']
data['FUVlum_err'] = FUVdata['luminosity_hostsub_err']
data['NUVlum'] = NUVdata['luminosity_hostsub']
data['NUVlum_err'] = NUVdata['luminosity_hostsub_err']
data['color_lum'] = data['FUVlum'] - data['NUVlum']
data['color_lum_err'] = np.sqrt(data['FUVlum_err']**2 + data['NUVlum_err']**2)
data['color_ratio'] = data['FUVlum'] / data['NUVlum']
data['color_ratio_err'] = data['color_ratio'] * np.sqrt((data['FUVlum_err']/data['FUVlum'])**2 + (data['NUVlum_err']/data['NUVlum'])**2)
# print(data)

ratio = data[pd.notna(data['color_ratio'])]['color_ratio']
ratio_err = data[pd.notna(data['color_ratio'])]['color_ratio_err']
weighted_stats = DescrStatsW(ratio, weights=1/ratio_err**2, ddof=0)
mean_ratio = weighted_stats.mean
mean_ratio_err = weighted_stats.std
print(mean_ratio)
print(mean_ratio_err)

fig, ax = plt.subplots()
data = data[data.index > 0]
ax.errorbar(data.index, data['color_mag'], yerr=[data['color_mag_err_2'], data['color_mag_err_1']], linestyle='none', marker='o')
# ax.errorbar(data.index, data['color_lum'], yerr=data['color_lum_err'], linestyle='none', marker='o')
ax.set_xlabel('Rest frame time since discovery [days]')
ax.set_ylabel('FUV - NUV color [mag]')
# ax.set_ylabel('FUV - NUV luminosity [erg s$^{-1}$ Å$^{-1}$]')
ylim = ax.get_ylim()
ax.set_ylim((ylim[1], ylim[0]))
# plt.show()
plt.close()

# Estimate temperature
r = 0
temp = 10000 # K
lambda_eff = {'FUV': 1549, 'NUV': 2304.7} # observed wavelength
z_factor = 1/(1+sn_info.loc[sn, 'z'])
while r < mean_ratio:
    temp += 100
    bb = models.BlackBody(temperature=temp*u.K)
    FUV = bb(lambda_eff['FUV'] * z_factor * u.AA)
    NUV = bb(lambda_eff['NUV'] * z_factor * u.AA)
    r = FUV / NUV

print(temp)
print(r)
print(FUV)
print(NUV)