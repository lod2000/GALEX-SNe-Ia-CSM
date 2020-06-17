import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.ned import Ned

QUERY_RADIUS = 1. # arcmin

def main():
	
	name1, name2 = 'Aki', 'SB1997ez'
	host1, host2 = 'CXO J221601.1​-173722', 'A082138+0325'
	ra1, dec1 = '22h16m01.077s', '-17d37m22.09s'
	ra2, dec2 = '8h21m38.13s', '3d25m10.5s'

	for name, host, ra, dec in zip([name1, name2], [host1, host2], [ra1, ra2], [dec1, dec2]):
		print('\nQuerying Ned for: %s' % name)
		ned_table = query_ned_loc(ra, dec, radius=QUERY_RADIUS, verb=1)
		ned_table2 = query_ned_name(host)

		print('Location query results:')
		print(ned_table)
		best3 = np.argsort(ned_table['Separation'])[:3]
		print('Best matches:')
		print(ned_table[best3])
		if ned_table2 is not None:
			print('Object name query results:')
			print(ned_table2)
		else:
			print('Object name query failed.')


def query_ned_loc(ra, dec, radius=1., verb=0):
	"""
	Query NED based on sky coordninates
	Inputs:
		ra, dec (float): sky coords in HHhMMmSS.Ss str format
		radius (float, optional): query radius in arcmin, default=1
		verb (int or bool, optional): verbose output? Default: False
	Outputs:
		ned_table: astropy table of query results
	"""

	#parse coords
	coord = SkyCoord(ra, dec)

	#send query
	if verb:
		print('\tsending query...')
	results = Ned.query_region(coord, radius=radius*u.arcmin)
	if verb:
		print('\tcomplete.')
	return results

def query_ned_name(objname, verb=0):
	"""
	Query NEd based on an object name (e.g., host galaxy name)
	Inputs:
		objname (str): name of object
		verb (int or bool, optional): vebrose output? default: False
	Outputs:
		ned_table: table of query results
	"""

	if verb:
		print('\tsending query...')
	try:
		results = Ned.query_object(objname)
		if verb:
			print('\tcomplete')
	except:
		if verb:
			print('Object name query failed for object: %s' % objname)
		results = None
	return results


if __name__=='__main__':
	main()