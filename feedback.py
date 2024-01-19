import numpy as np
import os
import sys
import warnings

from sklearn.ensemble import RandomForestClassifier
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.scenario.scenario import Scenario

from train import train as T
from train import eval_model
from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO#smac_hpo_facade#, Scenario
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import UNet as Model
from Config import config as Config
from AugBlock import augBlock
from gtaV_helper import GTA5
from cityscapes_helper import CitySpaceseDataset

torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled
sys.path.append(os.path.abspath('./Automold--Road-Augmentation-Library'))

RESULT_FOLDER = '/Result'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'

train_path=r"../GTA5/data"
test_path= r"../Cityscapes/"
#%% Global Vars
cfg = Config()

os.makedirs(TENSORBOARD_PREFIX, exist_ok=True)
os.makedirs(f'{RESULT_FOLDER}/{cfg.exp_name}', exist_ok=True)   

writer = SummaryWriter(log_dir=f'{TENSORBOARD_PREFIX}/{cfg.exp_name}')

data_train = GTA5(train_path)
data_test = CitySpaceseDataset(test_path)

feedback_runs = 5
    
#%%



def train(config: Configuration, seed: int = 0) -> float:
    aug = [config["aug1"],config["aug2"],config["aug3"],config["aug4"],config["aug5"]]
    params = [config["param1"],config["param2"],config["param3"],None,None]
    train_loader =  DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=False, generator=torch.Generator(device='cpu'))
    test_loader =  DataLoader(data_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, generator=torch.Generator(device='cpu'))

    model = Model()
    model.cpu()

    # init optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

    val_train = T(model, train_loader, optimizer, scheduler, writer, cfg=None, test_loader = None,aug=aug,params=params)

    print('[*] train ends... ')
    print(f'[*] Loss train: {val_train}')

    score_test = eval_model(model, test_loader,best_val=100, cfg=None)
    print('[*] Eval ends... ')
    print(f'[*] Score test: {score_test}')
    return 1 - score_test


configspace = ConfigurationSpace({"aug1": (0.0, 0.2),
                                  "aug2": (0.0, 0.2),
                                  "aug3": (0.0, 0.2),
                                  "aug4": (0.0, 0.2),
                                  "aug5": (0.0, 0.2),
                                  "param1": (0.01, 0.99),
                                  "param2": (0.01, 0.99),
                                  "param3": (0.01, 0.99)})

# Scenario object specifying the optimization environment
# scenario = Scenario(configspace, deterministic=True, n_trials=200)
scenario = Scenario({
    "run_obj": "quality",  # Optimize quality (alternatively runtime)
    "runcount-limit": feedback_runs,  # Max number of function evaluations (the more the better)
    "cs": configspace,
    "deterministic": True,
    "output_dir": Path(r".\smac3_output"),
})


# Use SMAC to find the best configuration/hyperparameters
smac = SMAC4HPO(scenario  = scenario,tae_runner = train)
incumbent = smac.optimize()
print(incumbent._values)
#%% Test best values
aug = [incumbent._values["aug1"],incumbent._values["aug2"],incumbent._values["aug3"],incumbent._values["aug4"],incumbent._values["aug5"]]
params = [incumbent._values["param1"],incumbent._values["param2"],incumbent._values["param3"],None,None]


train_loader =  DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=False, generator=torch.Generator(device='cpu'))
test_loader =  DataLoader(data_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, generator=torch.Generator(device='cpu'))

model = Model()
model.cpu()

# init optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

val_train = T(model, train_loader, optimizer, scheduler, writer, cfg=None, test_loader = None,aug,params)

print('[*] train ends... ')
print(f'[*] Loss train: {val_train}')

score_test = eval_model(model, test_loader,best_val=100, cfg=None)
print('[*] Eval ends... ')
print(f'[*] Score test: {score_test}')


#%% Test all value ranges
Cs = np.linspace(1000,10000,100)
for c in Cs:
    classifier = SVC(C=c)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    print("C= ",c, np.mean(scores))
# Get cost of default configuration
# default_cost = smac.validate(classifier.configspace.get_default_configuration())
# print(f"Default cost: {default_cost}")

# # Let's calculate the cost of the incumbent
# incumbent_cost = smac.validate(incumbent)
# print(f"Default cost: {incumbent_cost}")
#%% configuration space
cs = ConfigurationSpace({
    "myfloat": (0.1, 1.5),                # Uniform Float
    "myint": (2, 10),                     # Uniform Integer
    "species": ["mouse", "cat", "dog"],   # Categorical
})
#%% Target Function
