import os
from typing import List

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
from database import DatabaseBook, DatabasePage
from database.file_formats.pcgts import PageScaleReference, Point, SymbolType
from database.file_formats.pcgts.page import AdvancedSymbolClass, AdvancedSymbolColor

dataset_location = "/tmp/Graduale_Synopticum"

import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose(
    [
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(),

    ]
)


class MemoryClassifcationDataset(Dataset):
    def __init__(self, images, labels1, label2):
        self.images = images
        self.labels1 = labels1
        self.labels2 = label2

    def __getitem__(self, item):
        image_ = self.images[item]
        label1 = self.labels1[item]
        label2 = self.labels2[item]

        image_ = train_transforms(image=image_)["image"]

        return image_, label1, label2

    def __len__(self):
        return len(self.images)


def vgg_block_single(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def vgg_block_double(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class VGG11(nn.Module):
    def __init__(self, in_ch, num_classes1, num_classes2):
        super().__init__()

        self.conv_block1 = vgg_block_single(in_ch, 64)
        self.conv_block2 = vgg_block_single(64, 128)

        self.conv_block3 = vgg_block_double(128, 256)
        self.conv_block4 = vgg_block_double(256, 512)
        self.conv_block5 = vgg_block_double(512, 512)

        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes1)
        )
        self.fc_layers2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)
        y = self.fc_layers2(x)

        return x, y


class Net(nn.Module):
    def __init__(self, in_ch, num_classes1, num_classes2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes1)
        self.gc1 = nn.Linear(128 * 2 * 2, 120)
        self.gc2 = nn.Linear(120, 84)
        self.gc3 = nn.Linear(84, num_classes2)

    def forward(self, x):
        # print("123")
        # print(x.shape)

        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print(x.shape)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        z = F.relu(self.gc1(x))
        z = F.relu(self.gc2(z))
        z = self.gc3(z)
        return y, z


def train(images, labels1, labels2, path="/tmp/model.torch"):
    batchsize = 16
    dataset = MemoryClassifcationDataset(images, labels1, labels2)

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    adv_color_ = len(AdvancedSymbolColor)
    adv_class_ = len(AdvancedSymbolClass)

    network = Net(3, adv_class_, adv_color_)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(dataset_loader)
    # imagesb, labels1b, labels2b = next(dataiter)

    # show images
    classes1 = [x.name for x in AdvancedSymbolClass]
    classes2 = [x.name for x in AdvancedSymbolColor]
    # print(' '.join(f'{classes1[labels1b[j]]:5s}' for j in range(batchsize)))
    # print(' '.join(f'{classes2[labels2b[j]]:5s}' for j in range(batchsize)))
    # imshow(torchvision.utils.make_grid(imagesb))

    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels1, labels2 = data

            # print(inputs.shape)
            # print(labels1.shape)
            # print(labels2.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output1, output2 = network(inputs)
            loss1 = criterion1(output1, labels1)
            # print(loss1)
            loss2 = criterion2(output2, labels2)
            # print("l2")
            # print(loss2)

            loss = loss1 + loss2
            # loss = loss1
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(i)
            if i % 5 == 4 and epoch > 100 and False:  # print every 2000 mini-batches

                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
                running_loss = 0.0
                print(output1)
                print(' '.join(f'{torch.argmax(output1[j])}' for j in range(batchsize)))
                print(' '.join(f'{classes1[torch.argmax(output1[j])]:5s}' for j in range(batchsize)))
                print(' '.join(f'{classes1[labels1[j]]:5s}' for j in range(batchsize)))
                print("___")
                print(output2)

                print(' '.join(f'{classes2[torch.argmax(output2[j])]:5s}' for j in range(batchsize)))

                print(' '.join(f'{classes2[labels2[j]]:5s}' for j in range(batchsize)))
                print("__")
                print(' '.join(f'{classes1[labels1[j]]:5s}' for j in range(batchsize)))
                print(' '.join(f'{classes2[labels2[j]]:5s}' for j in range(batchsize)))
                imshow(torchvision.utils.make_grid(inputs))

                print("next")
    print('Finished Training')
    torch.save(network.state_dict(), path)


def test(images, labels1, labels2, model_path, pages):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    batchsize = 16
    dataset = MemoryClassifcationDataset(images, labels1, labels2)

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)
    adv_color_ = len(AdvancedSymbolColor)
    adv_class_ = len(AdvancedSymbolClass)
    classes1 = [x.name for x in AdvancedSymbolClass]
    classes2 = [x.name for x in AdvancedSymbolColor]
    network = Net(3, adv_class_, adv_color_)
    network.load_state_dict(torch.load(model_path))
    print("Eval")
    with torch.no_grad():
        for data in dataset_loader:
            images, labels1, label2 = data
            # calculate outputs by running images through the network
            output1, output2 = network(images)
            # the class with the highest energy is what we choose as prediction

            print(' '.join(f'{classes1[torch.argmax(output1[j])]:5s}' for j in range(batchsize)))

            print(' '.join(f'{classes2[torch.argmax(output2[j])]:5s}' for j in range(batchsize)))
            print("next")

            imshow(torchvision.utils.make_grid(images))


def extract_image_point_arount_center_point(image, center: Point, size):
    y = int(center.y)
    x = int(center.x)
    return image[y - size:y + size, x - size:x + size]


if __name__ == "__main__":
    import django


    def train_data(pages: List[DatabasePage], valid_symbols: List[SymbolType] = None):
        images = []
        labels_e_class = []
        labels_e_color = []
        if valid_symbols is None:
            valid_symbols = [SymbolType.NOTE, SymbolType.CLEF, SymbolType.ACCID]
        for page in pages:
            page_id = page.page
            print(page_id)
            # page.pcgts()
            lines = page.pcgts().page.all_music_lines()
            image_path = page.file("color_original").local_path()
            print(image_path)
            image_url = Image.open(image_path)
            image = np.array(image_url)
            for line in lines:
                for symbol in line.symbols:
                    if symbol.symbol_type not in valid_symbols:
                        continue
                    if symbol.missing:
                        continue
                    # limited gt on page 0003
                    if page_id == "0003" and (
                            symbol.advanced_class == symbol.advanced_class.normal and symbol.advanced_color == symbol.advanced_color.black):
                        continue
                    coord = page.pcgts().page.page_to_image_scale(symbol.coord, PageScaleReference.ORIGINAL)
                    patch = extract_image_point_arount_center_point(image, coord, 8)
                    images.append(patch)
                    labels_e_class.append(symbol.advanced_class.value)
                    labels_e_color.append(symbol.advanced_color.value)
                    # print(symbol)
                    # print(symbol.to_json())
                    # print(coord)
                    # from matplotlib import pyplot as plt
                    # plt.imshow(patch)
                    # plt.show()
        return images, labels_e_class, labels_e_color


    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings

    # get all books from database
    pages = DatabaseBook("Graduel_Syn23_1_24").pages()[1:3]

    images, labels_e_class, labels_e_color = train_data(pages, valid_symbols=[SymbolType.NOTE])
    train(images, labels_e_class, labels_e_color, "/tmp/model.torch")
    pages = DatabaseBook("Graduel_Syn23_1_24").pages()[3:4]
    images, labels_e_class, labels_e_color = train_data(pages)

    test(images, labels_e_class, labels_e_color, model_path="/tmp/model.torch", pages=pages)
