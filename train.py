import numpy as np
import torch, torchvision
import time, warnings
import os
from torch.optim import optimizer
from matplotlib import pyplot as plt
from utils import stat_cuda, Timer, VGGRegressionModel
warnings.filterwarnings("ignore") 

device = torch.device('cuda') # 'cuda' or 'cpu'
print('Running on', device)

'''Directory'''
basedir = 'fashion-MNIST/'
if not os.path.isdir(basedir):
    os.makedirs(basedir)

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



'''Set up train dataset, dataloader'''
dataset = torchvision.datasets.FashionMNIST(root = '.data', train=True, download=True, transform=torchvision.transforms.ToTensor())
batch_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)
images, _ = dataiter.next()
mean, std = images.mean(), images.std()
del(images, dataiter, dataloader, batch_size)
# mean, std = 0.2860, 0.3530

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
dataset = torchvision.datasets.FashionMNIST(root = '.data', train=True, download=True, transform=transform)

batch_size = 64
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)

'''Define network'''
custom_config = [32, 'M', 64, 'M']


'''Define model, optimizer, error function'''
model = VGGRegressionModel(custom_config).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()

epoch_init = -1
n_epochs = 2

'''Load model checkpoint'''
if epoch_init>=0:
    checkpoint = torch.load(basedir + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    del(checkpoint)
else:
    losses = []


'''Train network'''
iter_num = int(np.ceil(len(dataset)/batch_size))
with Timer('training phase'):
    for epoch in range(epoch_init+1, n_epochs):
        epoch_loss = torch.tensor([])
        for batch_index, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            predictions = model(inputs.to(device))
            l = criterion(predictions, labels.to(torch.long).to(device))
            l.backward()
            optimizer.step()
            epoch_loss = torch.cat((epoch_loss, torch.tensor(l.item()).unsqueeze(0)), dim=0)
            if batch_index%300 == 0:
                print(f'Loss = {l.item():.4f} at epoch {epoch} and batch {batch_index}/{iter_num}')
                stat_cuda()

        losses.append(epoch_loss.mean())

        '''Save model checkpoint'''
        torch.save({
                    'epoch': epoch,
                    'losses': losses,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, basedir + 'model_e%d.pt' % (epoch))

## Plot loss
plt.plot(losses[0:])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig(basedir + 'loss.png') 

'''Evaluate Network on testset'''
print('Evaluating on testset')
## Set up test dataset
dataset = torchvision.datasets.FashionMNIST(root = '.data', train=False, download=True, transform=torchvision.transforms.ToTensor())
batch_size = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)
images, _ = dataiter.next()
mean, std = images.mean(), images.std()
del(images, dataiter, dataloader, batch_size)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
dataset = torchvision.datasets.FashionMNIST(root = '.data', train=False, download=True, transform=transform)

batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(dataloader)


accuracy = 0
with torch.no_grad():
    prediction_list, label_list = [], []
    for idx in range(len(dataset)):
        input, img_id = dataset[idx]
        prediction_list.append(model(input.unsqueeze(0).to(device)).cpu().argmax().item())
        label_list.append(img_id)
        if label_list[idx]==prediction_list[idx]: accuracy+=1
accuracy /= len(dataset)
print(accuracy)