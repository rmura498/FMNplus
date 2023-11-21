import pandas as pd
import numpy as np
import os, math
import torch
import pickle

from Utils.metrics import compute_robust

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def evaluate_strategies(batch_size=10):

    sample_columns = [f'Sample{i}' for i in range(batch_size)]
    df_columns = ['Strategies - [Opt-Sch-Loss-Grad-Init]']
    BIG_DF = pd.DataFrame(columns=df_columns+sample_columns+['winner'])

    filenames = os.listdir('../experiments/')

    for i, filename in enumerate(filenames):
        splits = filename.split('.')
        strategy = splits[0]
        try:
            with open(os.path.join("../experiments/", filename), 'rb') as file:
                attack_data = pickle.load(file)
        except FileNotFoundError:
            continue

        distances = attack_data['distance']
        distances = [distance.tolist() for distance in distances]
        min_distances = []
        for j in range(batch_size):
            sample_distance = []
            for distance in distances:
                sample_distance.append(distance[j])
            min_dist = "{:.7f}".format(sample_distance[-1])
            min_distances.append(min_dist)


        BIG_DF.loc[i] = [strategy] + min_distances + [0]
    min_values = BIG_DF.min()
    with open("min_comparison_latex.txt", "w+") as file:
        latex_string = min_values.to_latex()
        file.writelines(latex_string)
    print(min_values)

    # Iterate through each column and update the 'winner' column
    for column in BIG_DF.columns[1:-1]:
        is_min = BIG_DF[column] == min_values[column]
        BIG_DF['winner'] += is_min.astype(int)

    # Display the updated DataFrame
    with open("comparison.csv", "w+") as file:
        file.writelines(BIG_DF.to_csv(index=False, float_format='%.7f'))
    with open("tuning_comparison_latex.txt", "w+") as file:
        latex_string = BIG_DF.to_latex()
        file.writelines(latex_string)


