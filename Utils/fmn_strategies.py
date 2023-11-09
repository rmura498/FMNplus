import itertools

loss = ['LL', 'CE', 'DLR']
optimizer = ['SGD', 'Adam']
scheduler = ['CALR', 'RLROP']
gradient_update = ['Normalization', 'Projection']


def fmn_strategies():
    # Create all possible combinations of elements from the lists
    combinations = list(itertools.product(loss, optimizer, scheduler, gradient_update))

    # Initialize an empty dictionary
    result_dict = {}

    # Assign a numeric value to each combination and create the dictionary
    for i, combo in enumerate(combinations):
        key = f'{i}'
        result_dict[key] = {
            'loss': combo[0],
            'optimizer': combo[1],
            'scheduler': combo[2],
            'gradient_strategy': combo[3],
        }

    return result_dict