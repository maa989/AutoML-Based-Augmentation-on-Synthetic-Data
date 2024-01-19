import numpy as np

from sklearn.ensemble import RandomForestClassifier
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from sklearn.neural_network import MLPClassifier
from train import main

#%% example
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO#smac_hpo_facade#, Scenario
from smac.scenario.scenario import Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from pathlib import Path
iris = datasets.load_diabetes()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
# scenario = Scenario(configspace, deterministic=True, n_trials=200)
scenario = Scenario({
    "run_obj": "quality",  # Optimize quality (alternatively runtime)
    "runcount-limit": 100,  # Max number of function evaluations (the more the better)
    "cs": configspace,
    "deterministic": True,
    "output_dir": Path(r"D:\USC\EE641\Project\EE641-Project\smac3_output"),
})


# Use SMAC to find the best configuration/hyperparameters
smac = SMAC4HPO(scenario  = scenario,tae_runner = train)
incumbent = smac.optimize()

#%%
import warnings
warnings.filterwarnings("ignore")
C = incumbent._values["C"]
classifier = SVC(C=C)
scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
print("IDEAL: ", np.mean(scores))

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
def train(self, config: cs, seed: int) -> float:
    model = MLPClassifier(learning_rate=config["learning_rate"])
    model.fit(...)
    accuracy = model.validate(...)

    return 1 - accuracy  # SMAC always minimizes (the smaller the better)

#%% Scenario
scenario = Scenario(
    configspace=cs,
    output_directory="your_output_directory",
    walltime_limit=120,  # Limit to two minutes
    n_trials=500,  # Evaluated max 500 trials
    n_workers=8  # Use eight workers
)
#%% Facade

#%%

X_train, y_train = np.random.randint(2, size=(20, 2)), np.random.randint(2, size=20)
X_val, y_val = np.random.randint(2, size=(5, 2)), np.random.randint(2, size=5)


def train_random_forest(config):
    model = RandomForestClassifier(max_depth=config["depth"])
    model.fit(X_train, y_train)

    # Define the evaluation metric as return
    return 1 - model.score(X_val, y_val)


if __name__ == "__main__":
    # Define your hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformIntegerHyperparameter("depth", 2, 100))

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 10,  # Max number of function evaluations (the more the better)
        "cs": configspace,
    })

    smac = SMAC4BB(scenario=scenario, tae_runner=train_random_forest)
    best_found_config = smac.optimize()
