import importlib
import sys
sys.path.insert(0, r'c:\Users\Moore Lab\Sharif\realtime-rdk')

import saccade

print('Calling genv reward callback (if set)')
try:
    if hasattr(saccade, 'genv') and getattr(saccade.genv, '_reward_callback', None) is not None:
        saccade.genv._reward_callback()
    else:
        print('genv reward callback not set; calling give_reward directly')
        saccade.give_reward()
except Exception as e:
    print('Exception while invoking reward callback:', e)
    import traceback
    traceback.print_exc()
