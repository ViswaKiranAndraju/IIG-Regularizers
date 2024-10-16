import os
import importlib
from datetime import datetime
from collections import defaultdict
import yaml
import pickle
import random
import math
import numpy as np
import pyspiel
import ray

# Import the BalancedFTRL class from its module
from agents.balanced_ftrl import BalancedFTRL  # Update this line to the correct import based on your directory structure

# Function to sample from weights
def sample_from_weights(population, weights):
    return random.choices(population, weights=weights)[0]

# Function to compute log(sum(exp(x)))
def compute_log_sum_from_logit(logit, mask):
    logit_max = logit.max(initial=-np.inf, where=mask)
    return math.log(np.sum(np.exp(logit - logit_max, where=mask), where=mask)) + logit_max

# Function to dynamically import classes
def get_class(class_name):
    module_names = class_name.split('.')
    module = ".".join(module_names[:-1])
    return getattr(importlib.import_module(module), module_names[-1])

class ExperimentGenerator(object):
    def __init__(self, description, game_names, agents, save_path,
                 global_init_kwargs=None, global_training_kwargs=None,
                 tuning_parameters=None, n_simulations=1):
        self.description = description
        self.game_names = game_names
        self.n_simulations = n_simulations
        self.global_init_kwargs = global_init_kwargs if global_init_kwargs else {}
        self.training_kwargs = global_training_kwargs if global_training_kwargs else {}
        self.tuning_parameters = tuning_parameters
        self.tuned_rates = None
        self.save_path = os.path.join(save_path, description)

        # Build the agent constructors
        self.dict_agent_constructor = {}
        self.dict_agent_kwargs = {}
        self.agent_names = []
        for agent_config_path in agents:
            agent_config = yaml.load(open(agent_config_path, 'r'), Loader=yaml.FullLoader)
            agent_class_name = agent_config['agent_class']
            agent_class = get_class(agent_class_name)
            agent_kwargs = agent_config['init_kwargs']
            if self.global_init_kwargs:
                agent_kwargs.update(self.global_init_kwargs)
            agent_name = agent_kwargs['name']
            self.agent_names.append(agent_name)
            self.dict_agent_kwargs[agent_name] = agent_kwargs
            self.dict_agent_constructor[agent_name] = agent_class

    def save_results(self, results, game_name, agent_name):
        now = datetime.now().strftime("%d-%m__%H:%M")
        save_path = os.path.join(self.save_path, game_name, agent_name, now + '.pickle')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as _f:
            pickle.dump(results, _f)

    def load_results(self):
        dict_results = {}
        for game_name in self.game_names:
            dict_results[game_name] = {}
            for agent_name in self.agent_names:
                save_path = os.path.join(self.save_path, game_name, agent_name)
                list_res = os.listdir(save_path)
                latest_res = max(list_res)
                save_path = os.path.join(save_path, latest_res)
                with open(save_path, 'rb') as _f:
                    dict_results[game_name][agent_name] = pickle.load(_f)
        return dict_results

    def run(self):
        list_tasks = []
        for game_name in self.game_names:
            for agent_name in self.agent_names:
                for _ in range(self.n_simulations):
                    base_constant = self.tuned_rates[game_name][agent_name] if self.tuned_rates else 1.0
                    list_tasks.append([
                        self.dict_agent_constructor[agent_name],
                        self.dict_agent_kwargs[agent_name],
                        game_name,
                        base_constant,
                        self.training_kwargs
                    ])
        ray.init()
        result_ids = [fit_agent.remote(*task) for task in list_tasks]
        results = ray.get(result_ids)
        ray.shutdown()
        print('Finished!')
        idx = 0
        for game_name in self.game_names:
            for agent_name in self.agent_names:
                final_results = defaultdict(list)
                for _ in range(self.n_simulations):
                    res = results[idx]
                    for key, value in res.items():
                        final_results[key].append(value)
                    idx += 1
                for key in final_results.keys():
                    final_results[key] = np.array(final_results[key]) if key != 'step' else final_results[key][0]
                self.save_results(final_results, game_name, agent_name)

    def tune_rates(self):
        lowest_multiplier = self.tuning_parameters['lowest_multiplier']
        highest_multiplier = self.tuning_parameters['highest_multiplier']
        size_grid_search = self.tuning_parameters['size_grid_search']
        log_step = (math.log(highest_multiplier) - math.log(lowest_multiplier)) / (size_grid_search - 1)
        base_constants = [lowest_multiplier * math.exp(i * log_step) for i in range(size_grid_search)]
        tuning_kwargs = self.training_kwargs.copy()
        tuning_kwargs.update({
            'record_exploitabilities': True,
            'number_points': None,
            'log_interval': self.global_init_kwargs['budget'],
            'record_current': False
        })
        list_tasks = []
        for game_name in self.game_names:
            for agent_name in self.agent_names:
                for base_constant in base_constants:
                    list_tasks.append([
                        self.dict_agent_constructor[agent_name],
                        self.dict_agent_kwargs[agent_name],
                        game_name,
                        base_constant,
                        tuning_kwargs
                    ])
        ray.init()
        result_ids = [fit_agent.remote(*task) for task in list_tasks]
        results = ray.get(result_ids)
        ray.shutdown()
        print("Finished tuning!")
        idx = 0
        self.tuned_rates = {}
        for game_name in self.game_names:
            self.tuned_rates[game_name] = {}
            for agent_name in self.agent_names:
                best_gap = float('inf')
                for base_constant in base_constants:
                    gap = results[idx].get('average')[0]
                    if self.tuned_rates[game_name].get(agent_name) is None or best_gap > gap:
                        best_gap = gap
                        self.tuned_rates[game_name][agent_name] = base_constant
                    idx += 1
        print("Best multipliers:")
        print(self.tuned_rates)

# Remote function for fitting agents
@ray.remote
def fit_agent(agent_constructor, agent_kwargs, game_name, base_constant, training_kwargs):
    agent_kwargs['game'] = pyspiel.load_game(game_name)
    agent_kwargs['base_constant'] = base_constant
    agent = agent_constructor(**agent_kwargs)
    print(f'Train {agent.name} on {game_name}')
    return agent.fit(**training_kwargs)

# Main function to run the experiment
def main():
    description = "experiment_1"  # Give a suitable description
    game_names = ["kuhn_poker"]  # Replace with your game name(s)
    agents = ["path/to/agent_config.yaml"]  # Update with your agent config paths
    save_path = "./results"  # Path to save results
    global_init_kwargs = {
        "budget": 100  # Example budget; adjust as needed
    }
    global_training_kwargs = {
        # Add any training kwargs here
        "num_iterations": 1000,
        "log_interval": 100,
    }
    tuning_parameters = {
        "lowest_multiplier": 0.1,
        "highest_multiplier": 2.0,
        "size_grid_search": 10
    }

    # Create the experiment generator
    exp_gen = ExperimentGenerator(
        description=description,
        game_names=game_names,
        agents=agents,
        save_path=save_path,
        global_init_kwargs=global_init_kwargs,
        global_training_kwargs=global_training_kwargs,
        tuning_parameters=tuning_parameters,
        n_simulations=10
    )

    # Run the experiment
    exp_gen.run()
    # Optionally tune rates
    # exp_gen.tune_rates()

if __name__ == "__main__":
    main()
