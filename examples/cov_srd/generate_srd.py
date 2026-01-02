from augur.generate import generate
from pathlib import Path
import os
path = Path(__file__).parent
fpath = os.path.join(path, 'config_test.yml')
likelihood, sacc_data, tools = generate(fpath, return_all_outputs=True, force_read=True)
opath = os.path.join(path, 'srd.fits')
sacc_data.save_fits(opath, overwrite=True)

exit()
