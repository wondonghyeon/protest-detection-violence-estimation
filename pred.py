"""
created by: Donghyeon Won
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.models as models

from util import ProtestDatasetEval, modified_resnet50


def eval_one_dir(img_dir, model):
        """
        return model output of all the images in a directory
        """
        # make dataloader
        dataset = ProtestDatasetEval(img_dir = img_dir)
        data_loader = DataLoader(dataset,
                                num_workers = args.workers,
                                batch_size = args.batch_size)
        # load model

        outputs = []
        imgpaths = []
        # need to change tqdm thing.. (consider the last iter)
        with tqdm(total=len(os.listdir(img_dir))) as pbar:
            for i, sample in enumerate(data_loader):
                imgpath, input = sample['imgpath'], sample['image']
                if args.cuda:
                    input = input.cuda()

                input_var = Variable(input)
                output = model(input_var)
                outputs.append(output.cpu().data.numpy())
                imgpaths += imgpath
                pbar.update(args.batch_size)


        df = pd.DataFrame(np.zeros((len(os.listdir(img_dir)), 13)))
        df.columns = ["imgpath", "protest", "violence", "sign", "photo",
                      "fire", "police", "children", "group_20", "group_100",
                      "flag", "night", "shouting"]
        df['imgpath'] = imgpaths
        df.iloc[:,1:] = np.concatenate(outputs)
        df.sort_values(by = 'imgpath', inplace=True)
        return df

def main():

    # load trained model
    print("*** loading the model from {model}***".format(model = args.model))
    model = modified_resnet50()
    if args.cuda:
        model = model.cuda()
    with open(args.model) as f:
        model.load_state_dict(torch.load(f)['state_dict'])
    print("*** calculating the model outputs of the images in {img_dir}***"
            .format(img_dir = args.img_dir))

    # calculate output
    df = eval_one_dir(args.img_dir, model)

    # write csv file
    df.to_csv(args.output_csvpath, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",
                        type=str,
                        required = True,
                        help = "image directory to calculate output"
                        "(the directory must contain only image files)"
                        )
    parser.add_argument("--output_csvpath",
                        type=str,
                        default = "result.csv",
                        help = "path to output csv file"
                        )
    parser.add_argument("--model",
                        type=str,
                        required = True,
                        help = "model path"
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 16,
                        help = "batch size",
                        )
    args = parser.parse_args()

    main()
