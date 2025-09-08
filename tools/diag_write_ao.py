import traceback
from nidaqmx.system import System
import nidaqmx
from nidaqmx.constants import VoltageUnits

print('Devices:', [d.name for d in System.local().devices])

try:
    t = None
    try:
        t = nidaqmx.Task(task_name='diag_task')
    except TypeError:
        t = nidaqmx.Task()
    print('Created task object, adding ao channel...')
    t.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=0.0, max_val=10.0, units=VoltageUnits.VOLTS)
    print('Channel added, writing 1.0 V')
    t.write(1.0)
    print('Write succeeded, writing 0')
    t.write(0.0)
    t.close()
    print('Closed task cleanly')
except Exception as e:
    print('Caught exception during diag task operation:', repr(e))
    traceback.print_exc()
