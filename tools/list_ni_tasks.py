import traceback

try:
    from nidaqmx.system import System
    s = System.local()
    try:
        task_names = [t.name for t in s.tasks]
    except Exception as e:
        task_names = f'Error listing tasks: {e}'
    try:
        device_names = [d.name for d in s.devices]
    except Exception as e:
        device_names = f'Error listing devices: {e}'

    print('NI-DAQ System tasks:', task_names)
    print('NI-DAQ Devices:', device_names)

    print('\nTask details:')
    for t in s.tasks:
        try:
            chans = [ch.physical_channel_name for ch in t.channels]
        except Exception as e:
            chans = f'unavailable: {e}'
        print(f"Task: {t.name}  Channels: {chans}")

    # Also show tasks that are exported by nidaqmx.task.Task if possible
    try:
        import nidaqmx
        print('\nOpen tasks via nidaqmx.Task.get_task_names():')
        try:
            print(nidaqmx.Task.get_task_names())
        except Exception as e:
            print('get_task_names error:', e)
    except Exception:
        pass

except Exception:
    print('nidaqmx error:')
    traceback.print_exc()
