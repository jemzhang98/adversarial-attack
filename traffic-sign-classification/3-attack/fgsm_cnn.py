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


def imshow(img, title):
    npimg = img.cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def fgsm_targeted_attack(model, img_input, origin_label, target_label):
    transform = transforms.Compose(
        [transforms.Normalize((1/0.440985, 1/0.390349, 1/0.438721), (1/0.248148, 1/0.230837, 1/0.237781)),
         transforms.ToPILImage])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imshow(torchvision.utils.make_grid(img_input, normalize=True), args.test_file)

    target_label_as_var = Variable(torch.from_numpy(np.asarray([target_label])))
    target_label_as_var = target_label_as_var.to(device)
    loss = nn.CrossEntropyLoss()
    img_input.requires_grad = True

    for i in range(1000):
        print('Iteration:', str(i))
        img_input.grad = None
        output = model(img_input)
        cost = loss(output, target_label_as_var)
        cost.backward()
        img_input.data = img_input.data - 0.3 * torch.sign(img_input.grad.data)

        attack_output = model(img_input)

        index = int(torch.argmax(attack_output))
        print('The prediction is: ' + labels[index])

        if int(torch.argmax(attack_output)) == target_label:
            imshow(torchvision.utils.make_grid(img_input, normalize=True).detach(), "targeted image")
            # orig_img = transform(img_input[0])
            # orig_img = np.array(img_input.detach()[0])
            # orig_img = orig_img.transpose(1, 2, 0)
            # cv2.imwrite('Data/Attack/targeted.png', orig_img)
            break
    return 1


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: StnCnn = StnCnn.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),
         transforms.Normalize((0.440985, 0.390349, 0.438721), (0.248148, 0.230837, 0.237781))])

    img = Image.open(args.test_file)
    img_rgb = transform(img.convert("RGB"))
    img_input = img_rgb.view(1, 3, 48, 48).to(device)
    fgsm_targeted_attack(model, img_input, 12, 2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--test_file', type=str, required=False)
    args = parser.parse_args()
    args.checkpoint = "lightning_logs/version_1/checkpoints/epoch=22.ckpt"
    args.test_file = "Data/Test/0010.png"
    main(args)
