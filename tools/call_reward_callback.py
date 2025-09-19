import importlib
import sys
sys.path.insert(0, r'c:\Users\Moore Lab\Sharif\realtime-rdk')

import gng_training

print('Calling genv reward callback (if set)')
try:
    if hasattr(gng_training, 'genv') and getattr(gng_training.genv, '_reward_callback', None) is not None:
        gng_training.genv._reward_callback()
    else:
        print('genv reward callback not set; calling give_reward directly')
        gng_training.give_reward()
except Exception as e:
    print('Exception while invoking reward callback:', e)
    import traceback
    traceback.print_exc()
