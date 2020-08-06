import numpy as np
import matplotlib.pyplot as plt
# import ipdb
from lmfit.models import GaussianModel
from scipy.interpolate import interp1d

W0 = 1000. #model wwavelength start
W1 = 3000. #model wavelength end
DW = 0.1 #model wavelength step

GALEX_EFF_AREA = np.pi*25.0**2. # cm2
HST_AREA = (1.2e2)**2. * np.pi # cm2


T0 = 0. #model time start
T1 = 3000. #model time end
DT = 0.1 #model time step

COLORS = { #for simple plotting
	'NUV':'g', 
	'FUV':'c',
	'F275W':'r'
}

def main(tstart, twidth, decay_rate):

	model1 = CSMmodel(tstart = tstart, twidth=twidth, decay_rate=decay_rate, scale=1)
	
	test = np.arange(250., 2500., 20)
	for z,m in zip([0.05, 0.25], ['.', 's']):
		f = model1(test, z)
		for name, fls in f.items():
			plt.plot(test, fls, label=name+' - z=%.2f' % z, color=COLORS[name], marker=m)

	plt.ylim(1e34, 1e38)
	plt.yscale('log')
	plt.xlim(250, 1500)
	plt.legend()
	plt.xlabel('time since max')
	plt.ylabel('Filter Luminoisity [erg/s/A]')
	plt.show()

	zvals = np.arange(0.01, 0.5, 0.01)
	vals = {
		'FUV':[],
		'NUV':[],
		'F275W':[]
	}

	for z in zvals:
		f = model1(400., z)
		for name, fl in f.items(): vals[name].append(fl)

	for name, color in zip(['F275W', 'NUV', 'FUV'], ['r', 'g', 'b']):
		plt.plot(zvals, vals[name], color+'.:', label=name)


	plt.ylim(0.9e36, 2e38)
	plt.yscale('log')
	plt.legend()

	plt.xlabel('Redshift')
	plt.ylabel('Filter Luminosity')

	plt.show()



class CSMmodel:
	def __init__(self, tstart, twidth, decay_rate, scale=0.5, vwidth=2000):
		"""
		Initializes a CSM model with input variables:
			tstart: Days after peak that ejecta impacts CSM 
			twidth: length of the light curve plateau in days
			decay_rate: decay rate per 100 days after plateau ends - see Graham paper
			scale: multiplicative scale factor applied to the CSM model to indicative brighter/fainter CSM interaction
			vwidth: velocity width of the CSM emission lines in km/s, using 2000 based on typical emission lines of SNe Ia-CSM

		Attributes:
			wlarr: np.array of wavelength values for the Chevalier CSM model determined by W0,W1,DW
			chev_model: Class instance of Chev94Model with input scale and vwidth parameters
			time: time array used for model interpolation determined by T0,T1,DT
			tscale: scale factor applied to the CSM light curve after platuea regime has ended determined by decay_rate

		Methods (see indvidual methods for more info):
			___call___: generates a model CSM profile at input redshift and time and compute the resulting filter luminosity


		"""

		self.wlarr = np.arange(W0, W1, DW)

		self.chev_model = Chev94Model(vwidth=vwidth, scale=scale)

		self.time = np.arange(T0, T1, DT)
		self.tscale = np.zeros_like(self.time)
		idx1 = (self.time >= tstart) * (self.time < (tstart + twidth))
		self.tscale[idx1] = 1.
		idx2 = (self.time >= (tstart + twidth))
		self.tscale[idx2] = (decay_rate)**((self.time[idx2]-(tstart+twidth))/100.)



	def __call__(self, t, z, plot_spec = False):
		"""
		Computes the filter luminosity for the CSMmodel at a given epoch and redshift
		Arguments:
			t (float or np.array): time of observations to return filter lumnoisites
			z (float): redshift
			plot_spec (bool, optional): plot the redshifted+scaled spectrum before returning filter luminosities? Default: False

		Returns:
			filter_lum (dict): Luminosities per filter for this CSM model at input times t

		"""


		#only running it at 1yr epoch
		init_spec = self.chev_model.gen_model(366.) # erg /s / A

		#shift to redshift		
		wl_obs = self.wlarr * (1.+z)

		#compute fluxes
		tscale = np.interp(t, self.time, self.tscale)

		if plot_spec:
			plt.plot(self.wlarr, init_spec, 'k-', label='original')
			if isinstance(t, float):
				plt.plot(wl_obs, tscale*init_spec, label='shifted+scaled (t=%.1f)' % t)
			else:
				for _t, ts in zip(t, tscale):
					plt.plot(wl_obs, ts*init_spec, label='shifted+scaled (t=%.1f)' % _t)
			plt.legend()
			plt.show()

		fluxes = {
			'FUV':compute_fuv(wl_obs, init_spec)*tscale,
			'NUV':compute_nuv(wl_obs, init_spec)*tscale,
			'F275W': compute_f275w(wl_obs, init_spec)*tscale
		}

		return fluxes


def compute_nuv(wl, flux):
	# read resp table
	det_wl, det_fl = np.genfromtxt('NUV.resp', unpack=True, dtype=float)
	det_fl /= GALEX_EFF_AREA
	filt = interp1d(det_wl, det_fl, kind='slinear', bounds_error=False, fill_value=0.)
	obs_fl = filt(wl) * flux # ergs/s/A
	return obs_fl.sum()

def compute_fuv(wl, flux):
	#read resp table
	det_wl, det_fl = np.genfromtxt('FUV.resp', unpack=True, dtype=float)
	det_fl /= GALEX_EFF_AREA
	filt = interp1d(det_wl, det_fl, kind='slinear', bounds_error=False, fill_value=0.)
	obs_fl = filt(wl) * flux
	return obs_fl.sum()

def compute_f275w(wl, flux):

	#read resp table, already in throughput so no need to scale
	det_wl, det_fl = np.genfromtxt('F275W.resp', unpack=True, dtype=float)
	filt = interp1d(det_wl, det_fl, kind='slinear', bounds_error=False, fill_value=0.)
	obs_fl = filt(wl) * flux
	return obs_fl.sum()


class Chev94Model:
	def __init__(self, vwidth=2000., scale=1.):
		"""
		CSM model based on Chevalier+Fransson 1994 paper derived for Type II SNe
		This ignores continuum contributions and only computes the expected line emission
		Arguments:
			vwidth (float): velocity width of the line profiles in km/s, default=2000
			scale (float): multiplicative scale factor to shift the CSM model bright/fainter, default=1.
		"""

		self.fname = 'chev_model.dat'
		self.times = np.array([1., 2., 5., 10., 17.5, 30.])*365.25
		self.model_data = {}
		with open(self.fname, 'r') as f:
			for (name, wl, fl1, fl2, fl5, fl10, fl17, fl30) in [line.strip().split() for line in f.readlines() if not line.startswith('#')]:
				#print(name)
				if name == 'Hbeta':
					coeffs = np.array([float(fl)*1e36*scale for fl in [fl1, fl2, fl5, fl10, fl17, fl30]])
					continue
				linelum = np.array([float(fl) for fl in [fl1,fl2,fl5,fl10,fl17,fl30]])*coeffs
				model = LineModel(float(wl), self.times, linelum)
				self.model_data[name] = model

	def gen_model(self, t):
		wl = np.arange(W0, W1, DW)
		fl = np.zeros_like(wl)
		for name, model in self.model_data.items():
			fl += model(t)
		return fl

class LineModel:
	def __init__(self, wl, times, linelum, vwidth=500.):
		"""
		Model for a given emission line profile
		Arguments:
			wl: wl array used for generating the model
			times: input times for corresponding line luminosities
			linelum: integrated luminosity of the line at input times
			vwidth: velocity width of the lines in km/s
		"""

		self.wl = wl
		self.vwidth = vwidth
		self.times = times
		self.linelum = linelum
		spectrum = gen_spectrum(wl, vwidth)
		lum = np.array([lum*spectrum for lum in self.linelum])

		self.interper = interp1d(self.times, lum.T, kind='quadratic', bounds_error=False, fill_value='extrapolate')

	def __call__(self, t):
		if t < 0: raise ValueError
		else:
			return self.interper(t)



def gen_spectrum(wl, vwidth):
	arr = np.arange(W0, W1, DW)
	gauss = GaussianModel()
	fl = gauss.eval(x=arr, center=wl, sigma=wl*vwidth/3e5, amplitude=1.)
	return fl



if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--tstart', '-s', help='When the ejecta impacts the CSM, days after max', default=300., type=float)
	parser.add_argument('--twidth', '-w', help='plateau width [days]', default=200., type=float)
	parser.add_argument('--decay-rate', '-D', help='fractional decay rate per 100 days', default=0.3, type=float)

	args = parser.parse_args()
	main(args.tstart, args.twidth, args.decay_rate)