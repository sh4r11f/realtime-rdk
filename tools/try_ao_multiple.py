import traceback
from nidaqmx.system import System
import nidaqmx
from nidaqmx.constants import VoltageUnits

print('Devices:', [d.name for d in System.local().devices])
print('Existing tasks at start:', [t.name for t in System.local().tasks])

persistent = None
try:
    try:
        persistent = nidaqmx.Task(task_name='persistent_test')
    except TypeError:
        persistent = nidaqmx.Task()
    persistent.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=0.0, max_val=10.0, units=VoltageUnits.VOLTS)
    print('Persistent task created:', getattr(persistent, 'name', '<unnamed>'))
    persistent.write(0.0)

    # Now try to create a second task using the same channel
    try:
        try:
            t2 = nidaqmx.Task(task_name='second_test')
        except TypeError:
            t2 = nidaqmx.Task()
        print('Created second task object, attempting to add channel...')
        t2.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=0.0, max_val=10.0, units=VoltageUnits.VOLTS)
        print('Second task channel added, attempting write...')
        t2.write(1.0)
        print('Second task write succeeded (unexpected)')
        t2.close()
    except Exception as e:
        print('Second task failed as expected:', repr(e))
        traceback.print_exc()

except Exception as e:
    print('Failed to create persistent task:', repr(e))
    traceback.print_exc()
finally:
    if persistent is not None:
        try:
            persistent.close()
            print('Persistent task closed')
        except Exception as e:
            print('Error closing persistent task:', e)
