from astroquery.exceptions import RemoteServiceError
from astroquery.ned import Ned

objname = '1WGA J2235.3-2557'

print('Querying object %s...' % objname)
ned_table = Ned.query_object(objname)
print(ned_table)

print('\nQuerying redshifts')
try:
    redshifts = Ned.get_table(objname, table='redshifts')
    print(redshifts)
except RemoteServiceError as e:
    print(e)

print('\nQuerying references...')
try:
    references = Ned.get_table(objname, table='references')
    print(references)
except RemoteServiceError as e:
    print(e)

print('\nQuerying notes...')
try:
    notes = Ned.get_table(objname, table='object_notes')
    print(ned_table)
except RemoteServiceError as e:
    print(e)