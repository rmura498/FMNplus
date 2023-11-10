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


    """
        distances_flat.sort()
        robust_per_iter.sort(reverse=True)
        idx = find_nearest(distances_flat, 8/255)
        

        distances = compute_best_distance(best_adv, inputs)
        distances = np.array(distances)
        acc_distances = np.linspace(0, 0.2, 500)
        robust = np.array([(distances > a).mean() for a in acc_distances])

        idx = find_nearest(distances, 8 / 255)

        std_robust = robust[idx]

        model_id = models_ids[model]

        tune_conf = BIG_DF.loc[(BIG_DF['Optimizer'] == opt_conversion[f'{optimizer}']) &
                               (BIG_DF['Scheduler'] == sch_conversion[f'{scheduler}']) &
                               (BIG_DF['Loss'] == loss)]
        robust_values = np.full(len(models_ids), None)
        robust_values[int(model_id[-1])] = std_robust

        if tune_conf.empty:
            values = [opt_conversion[f'{optimizer}'], sch_conversion[f'{scheduler}'], loss]

            values.extend(robust_values)
            new_series = pd.Series(dict(zip(df_columns, values)))

            BIG_DF = pd.concat([BIG_DF, new_series.to_frame().T], ignore_index=True)
        else:
            BIG_DF.loc[(BIG_DF['Optimizer'] == opt_conversion[f'{optimizer}']) &
                       (BIG_DF['Scheduler'] == sch_conversion[f'{scheduler}']) &
                       (BIG_DF['Loss'] == loss), [f'{model_id}']] = std_robust

    BIG_DF['Mean'] = BIG_DF.iloc[:, 3:11].mean(axis=1)
    BIG_DF.sort_values(inplace=True, by=['Optimizer', 'Scheduler', 'Loss'],
                       ascending=['Optimizer', 'Scheduler', 'Loss'])

    print(BIG_DF)

    with open("tuning_comparison.csv", "w+") as file:
        file.writelines(BIG_DF.to_csv(index=False, float_format='%.2f'))

    with open("tuning_comparison_latex.txt", "w+") as file:
        latex_string = BIG_DF.to_latex()
        file.writelines(latex_string)
    """

evaluate_strategies(5)