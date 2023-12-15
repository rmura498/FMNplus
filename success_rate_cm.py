import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import EXP_DIRECTORY


def main(exp_folder=None):
    if exp_folder is None:
        exp_folder = EXP_DIRECTORY

    # read all the folders inside exp_folder
    # retrieve date and model id
    # read each pkl inside the folder
    #   retrieve single exp data from pkl name
    #   extract attack data from the pkl file
    #   extract SR for the experiment and store it in a dictionary like:
    #   { 'Base-Opt-Sch-Loss-EI(?)': [SR0, is_adv0], ..., 'FMNVec-Opt-Sch-EI(?)': [SR1, is_adv1] }
    #       are there other ways?, maybe a pandas dataframe ...
    #   we extract SR to not re-compute it, maybe we could also check if it is the same from is_adv

    # having the data of the exps for each single model
    #   compute the SRs Confusion Matrix (compute it all, later we may split it)
    #   here a numpy matrix will be fine of shape (len(conf_keys), len(conf_keys)), sr_cm[0][0]



if __name__ == '__main__':
    main()