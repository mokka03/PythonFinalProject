import torch
import time

def stat_cuda(msg=''):
    print('GPU memory usage ' + msg + ':')
    print('allocated: %dM (max %dM), cached: %dM (max %dM)'
          % (torch.cuda.memory_allocated() / 1024 / 1024,
             torch.cuda.max_memory_allocated() / 1024 / 1024,
             torch.cuda.memory_reserved() / 1024 / 1024,
             torch.cuda.max_memory_reserved() / 1024 / 1024))

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def get_vgg_layers(config, batch_norm=True):
    
    layers = []
    in_channels = 1
    
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(c), torch.nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace = True)]
            in_channels = c
            
    return torch.nn.Sequential(*layers)

class VGGRegressionModel(torch.nn.Module):
    def __init__(self, config):
        super(VGGRegressionModel, self).__init__()
        self.features = get_vgg_layers(config)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64*7*7, 4096),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(256, 10),
        )

    def forward(self, x):
        # print('before convolution: ', x.shape)
        x = self.features(x)
        # print('after convolution: ', x.shape)
        x = x.view(x.shape[0], -1)
        # exit('exited in VGGRegressionModel')
        x = self.classifier(x)
        return x