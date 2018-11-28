"""
Michael Samon
Created on 11-27-18
Description: Data Reader class that prepares the sudoku dataset for training
Dataset Source: https://www.kaggle.com/bryanpark/sudoku/version/3
"""

import numpy as np
import os


class DataReader:

    # set folder path upon class initialization
    def __init__(self, repo_path):
        self.BASE_DIR = repo_path

    def get_data(self, save_npy=True):

        # check if '.npy' files are in base path
        base_files = os.listdir(self.BASE_DIR)

        if "y.npy" in base_files and "x.npy" in base_files:
            print("Loading data from local '.npy' files")
            x_path = os.path.join(self.BASE_DIR, "x.npy")
            y_path = os.path.join(self.BASE_DIR, "y.npy")

            x = np.load(x_path)
            y = np.load(y_path)
            return x, y

        filepath = os.path.join(self.BASE_DIR, "sudoku.csv")
        x = []
        y = []
        with open(filepath, 'r') as csv:
            raw_data = csv.readlines()
            for line in raw_data[1:]:
                line = line.strip("\n")
                xy = line.split(",")
                x.append([int(num) for num in xy[0]])
                y.append([int(num) for num in xy[1]])

        x = np.array(x)
        y = np.array(y)

        if save_npy:
            np.save("x.npy", x)
            np.save("y.npy", y)

        return x, y
