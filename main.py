import torch
import math
from Attacks.fmn import FMN
from Utils.load_model import load_dataset, load_model
from robustbench.utils import clean_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('Gowal2021Improving_R18_ddpm_100m', 'cifar10')
dataset = load_dataset('cifar10')
model.eval()
model.to(device)

batch = 100

dataset_frac = list(range(math.floor(len(dataset) * 0.5) + 1, len(dataset)))
dataset_frac = torch.utils.data.Subset(dataset, dataset_frac)
dl_test = torch.utils.data.DataLoader(dataset_frac, batch_size=batch,shuffle=False)
dl_test_iter = iter(dl_test)

attack = FMN(model, steps=20)
samples, labels = next(dl_test_iter)
adv_x = attack.forward(samples, labels)

clean_acc = clean_accuracy(model, samples, labels)
print("Clean Accuracy", clean_acc)
robust_acc = clean_accuracy(model, adv_x, labels)
print("Robust Accuracy", robust_acc)