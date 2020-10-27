import os
import numpy as np
import cv2
from torch import nn
import torch
from torch.autograd import Variable
from stn_cnn import StnCnn
from PIL import Image
from argparse import ArgumentParser
import torchvision.utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

labels = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
          'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']


def getindex(label):
    for i in range(len(labels)):
        if labels[i] == label:
            return i
    return 0


def imshow(img, title, save):
    npimg = img.cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def fgsm_untargeted_attack(model, file_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),
         transforms.Normalize((0.440985, 0.390349, 0.438721), (0.248148, 0.230837, 0.237781))])

    total = 0
    success = 0
    for file in os.listdir(file_folder):
        test_file = os.path.join(file_folder, file)
        print(test_file)
        total = total + 1

        img = Image.open(test_file)
        img_rgb = transform(img.convert("RGB"))
        img_input = img_rgb.view(1, 3, 48, 48).to(device)

        output = model(img_input)
        target_label = int(torch.argmax(output))

        # imshow(torchvision.utils.make_grid(img_input, normalize=True), args.test_file, False)

        target_label_as_var = Variable(torch.from_numpy(np.asarray([target_label])))
        target_label_as_var = target_label_as_var.to(device)
        loss = nn.CrossEntropyLoss()
        img_input.requires_grad = True

        for i in range(10):
            print('Iteration:', str(i))
            img_input.grad = None
            output = model(img_input)
            cost = loss(output, target_label_as_var)
            cost.backward()
            
            img_input.data = img_input.data + 0.3 * torch.sign(img_input.grad.data)

            attack_output = model(img_input)

            index = int(torch.argmax(attack_output))
            print('The prediction is: ' + labels[index])

            if int(torch.argmax(attack_output)) != target_label:
                success = success + 1
                # targeted_name = file_name + "_from_" + labels[origin_label] + "_to_" + labels[target_label] + ".png"
                # imshow(torchvision.utils.make_grid(img_input, normalize=True).detach(), targeted_name, True)
                break

    print(success)
    print(total)
    return 1


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: StnCnn = StnCnn.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    fgsm_untargeted_attack(model, args.test_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--test_folder', type=str, required=False)
    args = parser.parse_args()
    main(args)
