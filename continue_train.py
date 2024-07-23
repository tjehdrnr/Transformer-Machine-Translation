import sys
import os.path

import torch

from train import define_argparser
from train import main

def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for prev_key in vars(prev_config).keys():
        if prev_key not in vars(config).keys():
            # No such argument in current config, ignore that value.
            print('WARNING!!! Argument "--%s" is not found in current argument parser.' % prev_key +
            '\tIgnore saved value:', vars(prev_config)[prev_key])
    
    for key in vars(config).keys():
        if key not in vars(prev_config).keys():
            # No such argument in saved file. Use current value.
            print('WARNING!!! Argument "--%s" is not found in saved model.' % key +
            '\tUse current value:', vars(config)[key])
        elif vars(config)[key] != vars(prev_config)[key]:
            if '--%s' % key in sys.argv:
                # User changed argument value at this execution.
                print('WARNING!!! You changed value for argument "--%s".' % key +
                '\tUse current value:', vars(config)[key])
            else:
                vars(config)[key] = vars(prev_config)[key]
    
    return config


def continue_main(config, main):
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.load_fn):
        saved_data = torch.load(
            config.load_fn,
            map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
        )

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']

        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Can not find file %s' % config.load_fn)



if __name__ == '__main__':
    config = define_argparser(is_continue=True)
    continue_main(config, main)