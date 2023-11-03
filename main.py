import torch
import math
from Attacks.fmn import FMN
from Utils.load_model import load_dataset, load_model
from robustbench.utils import clean_accuracy
from Utils.plots import plot_distance

"""
FMN parametric strategy 
- the loss we're optimizing for the attack (CE, LL, DLR) 
- Initial point that we use for starting the attack 
- The optimizer that we use 
- the scheduler 
- Decide how we transform the gradient in each step. (normalization or linear projection into the step size)  

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('Gowal2021Improving_R18_ddpm_100m', 'cifar10')
dataset = load_dataset('cifar10')
model.eval()
model.to(device)

batch = 10

dataset_frac = list(range(math.floor(len(dataset) * 0.5) + 1, len(dataset)))
dataset_frac = torch.utils.data.Subset(dataset, dataset_frac)
dl_test = torch.utils.data.DataLoader(dataset_frac, batch_size=batch,shuffle=False)
dl_test_iter = iter(dl_test)
samples, labels = next(dl_test_iter)

steps=100

attack_LL = FMN(model, steps=steps, loss='LL')
attack_CE = FMN(model, steps=steps, loss='CE')
attack_DLR = FMN(model, steps=steps, loss='DLR')

adv_x_LL = attack_LL.forward(samples, labels)
adv_x_CE = attack_CE.forward(samples, labels)
adv_x_DLR = attack_DLR.forward(samples, labels)

clean_acc = clean_accuracy(model, samples, labels)
print("Clean Accuracy", clean_acc)
robust_acc = clean_accuracy(model, adv_x_LL, labels)
print("Robust Accuracy LL", robust_acc)

attack_LL_data = attack_LL.attack_data
distance_LL = attack_LL_data['distance']
distance_LL_0 = [distance[0].tolist() for distance in distance_LL]
epsilon_LL = attack_LL_data['epsilon']
epsilon_LL_0 = [epsilon[0].tolist() for epsilon in epsilon_LL]
loss_LL = attack_LL_data['loss']
loss_LL_0 = [loss[0].tolist() for loss in loss_LL]


robust_acc = clean_accuracy(model, adv_x_CE, labels)
print("Robust Accuracy CE", robust_acc)
attack_CE_data = attack_CE.attack_data
distance_CE = attack_CE_data['distance']
distance_CE_0 = [distance[0].tolist() for distance in distance_CE]
epsilon_CE = attack_CE_data['epsilon']
epsilon_CE_0 = [epsilon[0].tolist() for epsilon in epsilon_CE]
loss_CE = attack_CE_data['loss']
loss_CE_0 = [loss[0].tolist() for loss in loss_CE]


robust_acc = clean_accuracy(model, adv_x_DLR, labels)
print("Robust Accuracy DLR", robust_acc)
attack_DLR_data = attack_DLR.attack_data
distance_DLR = attack_DLR_data['distance']
distance_DLR_0 = [distance[0].tolist() for distance in distance_DLR]
epsilon_DLR = attack_DLR_data['epsilon']
epsilon_DLR_0 = [epsilon[0].tolist() for epsilon in epsilon_DLR]
loss_DLR = attack_DLR_data['loss']
loss_DLR_0 = [loss[0].tolist() for loss in loss_DLR]

plot_distance(distance_LL_0, epsilon_LL_0, loss_LL_0, 'LL')
plot_distance(distance_CE_0, epsilon_CE_0, loss_CE_0, 'CE')
plot_distance(distance_DLR_0, epsilon_DLR_0, loss_DLR_0, 'DLR')


