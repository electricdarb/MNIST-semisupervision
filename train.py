import torch
from torchvision import transforms
from tqdm import tqdm

from torchvision import datasets
from torchvision.transforms import ToTensor

IMAGE_SIZE = (28, 28, 1)

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, laten_dims = 32):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 'same')                             
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 'same')   

        # fully connected layer, laten dims output 
        self.out = nn.Linear(32 * 7 * 7, laten_dims)

    def forward(self, x):
        """x is a 28 x28 mnist image, returns a laten space representtion"""
        # layer 1
        x = self.conv1(x)
        x = F.relu(x) 
        x = F.max_pool2d(x, kernel_size = 2) # 14 x 14

        # layer 2
        x = self.conv2(x)
        x = F.relu(x) 
        x = F.max_pool2d(x, kernel_size = 2) # 7 x 7

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        laten_space = F.tanh(self.out(x))

        return laten_space

# vmap to allow batching, increases parrelism 
@torch.vmap
def sim_loss(laten_space):
    """similarity loss, positive examples are drawn closer, negative are pushed apart
    args:
        laten_space: tensor shape: (transforms, batch, laten_dims)
        device: device to put these tensors on
    """
    # cosine similarity of positive pairs, easy. 
    pos_cossim = torch.sum(F.cosine_similarity(laten_space[0], laten_space[1], dim = -1))

    # cosine similarity of negative pairs, hard
    # get indices pairs of negative examples, no repeats (ei no combos)
    indices = torch.tensor([*range(laten_space.shape[1])])
    neg_combos = torch.combinations(indices)

    # negative sims between row 1 and other row 1 latens
    neg_cossim_row1 = torch.sum(F.cosine_similarity(laten_space[0][neg_combos[:, 0]], laten_space[0][neg_combos[:, 1]], dim = -1))
    # negative sims between row 2 and other row 2 latens
    neg_cossim_row2 = torch.sum(F.cosine_similarity(laten_space[1][neg_combos[:, 0]], laten_space[1][neg_combos[:, 1]], dim = -1))
    # negative sims between row 1 and 2
    neg_cossim_inter = torch.sum(F.cosine_similarity(laten_space[1][neg_combos[:, 0]], laten_space[1][neg_combos[:, 1]], dim = -1))
    # sum of all sims
    neg_cossim = neg_cossim_row1 + neg_cossim_row2 + neg_cossim_inter
    
    # calculate nce 
    nce = - torch.log(torch.exp(pos_cossim) / (torch.exp(pos_cossim) + torch.exp(neg_cossim)))

    return nce    

if __name__ == "__main__":

    # load data and create dataloader
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    model = CNN()

    # keep this at 4, defines how many images each image is compared to
    TRANSFORMS = 4

    BATCH_SIZE = 16

    assert BATCH_SIZE % TRANSFORMS == 0, f'For vectorization purposes batch size be divisible by tranforms, instead {BATCH_SIZE=}, {TRANSFORMS=}'
    
    train_loader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, shuffle = True, drop_last = True)

    test_loader = torch.utils.data.DataLoader(test_data, BATCH_SIZE)

    # define transforms
    transform_pipe = transforms.Compose([
        transforms.Normalize(0, 1),
        transforms.RandomAffine(degrees = 15, translate = (.1, .1), scale = (.9, 1.1)),
        transforms.GaussianBlur(3),
        transforms.Lambda(lambda image: image + torch.randn(image.shape) * .05) # add 5% random noise
    ])

    opt = torch.optim.SGD(model.parameters(), lr = 0.003)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    EPOCHS = 25

    device = torch.device('cpu')

    model.to(device)

    losses = []

    for epoch in range(EPOCHS):
        train_loss = 0 
        # each batch in train
        for images, _ in tqdm(train_loader):
            # load images to device
            images = images.to(device)

            # apply transforms, images will be shape (2 * batchsize, ...)
            images = torch.concat([transform_pipe(images) for _ in range(2)], axis = 0)
            
            # run model and calc loss
            laten_spaces = model(images)

            # desired shape: (BATCH_SIZE/TRANSFORMS, 2, TRANSFORMS, laten_dims)
            # reshape to (2, BATCH_SIZE/TRANSFORMS, TRANSFORMS, laten_dims)
            laten_spaces = torch.reshape(laten_spaces, (2, BATCH_SIZE // TRANSFORMS, TRANSFORMS, -1))
            # swap axises to (BATCH_SIZE/TRANSFORMS, 2, TRANSFORMS, laten_dims))
            laten_spaces = torch.transpose(laten_spaces, 0, 1)
            
            # calculate loss 
            loss = torch.sum(sim_loss(laten_spaces))

            # run backwards pass and update model params 
            loss.backward()
            opt.step()

            # zero opt state 
            opt.zero_grad()

            # modify loss 
            train_loss += loss.item()
            losses.append(loss.item())

        train_loss /= len(train_loader)
        scheduler.step()

        print(f"Epoch: {epoch}/{EPOCHS}, {train_loss = }")

    torch.save(model, './simclrtest2.pt')