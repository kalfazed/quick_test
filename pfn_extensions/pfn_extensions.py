import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions, ExtensionsManager

import time
import math


# manager.extend(...) also works

def custom_extension(manager: ExtensionsManager):
    print('Epoch-iteration: {}-{}'.format(manager.epoch, manager.iteration))

def main():
    max_epoch = 10
    iters_per_epoch = 938

    models = {}
    optimizers = []

    manager = ppe.training.ExtensionsManager(
            models, optimizers, max_epoch, iters_per_epoch=iters_per_epoch)

    manager.extend(extensions.LogReport())
    manager.extend(extensions.ProgressBar())
    manager.extend(extensions.PrintReport(['epoch', 'iteration', 'sin', 'cos']))
    manager.extend(custom_extension, trigger=(1, 'epoch'))

    for epoch in range(max_epoch):
        for i in range(iters_per_epoch):
            with manager.run_iteration():
                ppe.reporting.report({
                    'sin': math.sin(i * 2 * math.pi / iters_per_epoch),
                    'cos': math.cos(i * 2 * math.pi / iters_per_epoch),
                })
                time.sleep(0.001)

if __name__ == "__main__":
    main()
