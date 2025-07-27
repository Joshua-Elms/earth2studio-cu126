from earth2studio.models.px import SFNO
import datetime as dt
from utils_E2S import general
from pathlib import Path
import numpy as np
from earth2studio.io import XarrayBackend
from earth2studio.data import CDS
import earth2studio.run as run
import dotenv

dotenv.load_dotenv()

# load the model
package = SFNO.load_default_package()
model = SFNO.load_model(package)

# load the initial condition times
ic_date = dt.datetime.strptime("2000/01/01 00:00", "%Y/%m/%d %H:%M")

# interface between model and data
xr_io = XarrayBackend()

# get ERA5 data from the ECMWF CDS
data_source = CDS() 

# run the model for all initial conditions at once
ds = run.deterministic(
    time=np.atleast_1d(ic_date), 
    nsteps=1,
    prognostic=model,
    data=data_source,
    io=xr_io,
    device="cuda",
).root

print(ds)