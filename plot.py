import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
def plot(args):
    train_stat_name_list=["Training loss", "Training accuracy(EM)"]
    train_stat_dict = dict()
    with open(os.path.join(args.plot_folder, args.plot_file ), "r") as fp:
        for i, line in enumerate(fp.readlines()):
            stat_d = json.loads(line)
            train_stat_dict[train_stat_name_list[i]]=stat_d

    # print(train_stat_dict)
    # loss_epochs = train_stat_dict["training loss"].keys()
    # print(loss_epochs)

    for train_stat_name in train_stat_name_list:
        xs =  list(train_stat_dict[train_stat_name].keys())
        ys = [ float(train_stat_dict[train_stat_name][x]) for x in xs]

        xs = [args.start_epoch+int(x) for x in xs]
        plt.figure() # initiate new figure
        print("xs: ", xs)
        print("ys: ", ys)
        plt.plot( xs, ys)
        plt.xlabel("epochs")
        plt.ylabel("train_stat_name")
        ys_float = [float(y) for y in ys]

        # print(ys_float)
        # plt.yticks(np.arange(min(ys_float), max(ys_float), step= (max(ys_float)-min(ys_float))/10 ), np.arange(min(ys_float), max(ys_float), step= 10 ))
        # name = f"{}/{train_stat_name}.png"
        plt.savefig("{}/{}.png".format(args.plot_folder, train_stat_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--plot_folder", default="out/nq-bart-closed-qa")
    parser.add_argument("--plot_file", default="bart_bs130.json")
    parser.add_argument("--start_epoch", default=0, type=int)
    args = parser.parse_args()

    plot(args)