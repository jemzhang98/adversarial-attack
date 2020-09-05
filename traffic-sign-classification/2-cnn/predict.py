import json
import os
import torch
import torchvision.transforms as transforms
from argparse import ArgumentParser
from PIL import Image
from stn_cnn import StnCnn
from tqdm import tqdm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
              'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']

    model: StnCnn = StnCnn.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    result = {}

    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),
         transforms.Normalize((0.440985, 0.390349, 0.438721), (0.248148, 0.230837, 0.237781))])
    for filename in tqdm(os.listdir(args.test_dir)):
        img = Image.open(os.path.join(args.test_dir, filename))
        img_rgb = transform(img.convert("RGB"))
        img_input = img_rgb.view(1, 3, 48, 48).to(device)
        output = model(img_input)
        index = int(torch.argmax(output))
        result[filename] = labels[index]

    with open(os.path.join(args.output_dir, 'result.json'), 'w') as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)

