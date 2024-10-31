from PIL import Image
import torch
import sys
import os
import time

import matplotlib.pyplot as plt
from torch import nn
from torchsampler.imbalanced import torchvision
from torchvision import transforms
from tqdm import tqdm, trange



img_trans =  transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor()
    ])

img_paths = sys.argv[1:]

images = torch.empty(len(img_paths),3,256,256)

for i in range(len(img_paths)):
    img = Image.open(img_paths[i]).convert('RGB')
    print("Image Format: ",img.format)
    if img.format == "GIF":
        print("Total frames: ", img.n_frames)
    img = img_trans(img)
    img = img.unsqueeze(0)
    images[i] = img
    

model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features  
model.fc = torch.nn.Linear(num_ftrs, 2)

state_dict = torch.load('./out/06214629/e31.pt',weights_only=True)
model.load_state_dict(state_dict)

model = model.cuda()
model.eval()

X = []
y = []
for i in range(len(img_paths)):
    
    
    image = images[i].unsqueeze(0)
    image = image.cuda()
    
    outputs = model(image)

    pred = outputs.argmax(dim=1)

    prob = nn.Softmax(dim=1)(outputs)[0]

    X.append(prob[1].item())

    isNailong = prob[1] > 0.90

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'{os.path.basename(img_paths[i])}: {prob[1]*100:.1f}%, {"y" if isNailong else "n"}, {elapsed_time:.2f}')

# n, bins, patches = plt.hist(X,bins=40,density=True,cumulative=True)
# plt.plot(bins[:-1], n, linestyle = '--', lw = 2, color = 'r')
# plt.show()
