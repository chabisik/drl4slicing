"""
This is a boilerplate pipeline 'focus_trf_d_dqn'
generated using Kedro 1.0.0
"""

import pfrl
import torch
import numpy as np
from torch.optim import Adam
from drl4slicing.pipelines.utils.InfrastructureManager import InfrastructureManager
from drl4slicing.pipelines.utils.NSPRGenerator import NSPRGenerator
from drl4slicing.pipelines.utils.Environment import Environment
from drl4slicing.pipelines.utils.FocusModule import TRFFocusModule
from drl4slicing.pipelines.utils.Model import AttentionQFunction_D_DQN
from drl4slicing.pipelines.utils.lion_pytorch import Lion


#======================================================================
# Nodes to construct environments
#======================================================================

def construct_infrastructures_managers(infrastructure_params: dict):
    # Remove parameters that are not required by InfrastructureManager class
    infrastructure_id = infrastructure_params["choice"]
    train_infra_mnger = InfrastructureManager(**infrastructure_params[infrastructure_id])
    eval_infra_mnger = InfrastructureManager(**infrastructure_params[infrastructure_id])
    return train_infra_mnger, eval_infra_mnger


def construct_nsprs_generators(nspr_params: dict):
    # Remove parameters that are not required by NSPRGenerator class
    level = nspr_params["level"]
    train_nsprs_gen = NSPRGenerator(**nspr_params[level])
    eval_nsprs_gen = NSPRGenerator(**nspr_params[level])
    return train_nsprs_gen, eval_nsprs_gen


def construct_environments(train_infra_mnger, train_nsprs_gen, eval_infra_mnger, eval_nsprs_gen):
    train_environment = Environment(train_infra_mnger, train_nsprs_gen)
    eval_environment  = Environment(eval_infra_mnger, eval_nsprs_gen)
    return train_environment, eval_environment


#======================================================================
# Nodes to construct agent
#======================================================================

def construct_focus_module(focus_params: dict):
    return TRFFocusModule(**focus_params)


def determine_agent_action_space_size(focus_params: dict, infrastructure_params: dict):
    if focus_params["active"]:
        # When focus module is active, the DRL algorithm's action space is reduced to the degree of focus
        n_actions = focus_params["degree"]
    else:
        infrastructure_id = infrastructure_params["choice"]
        n_actions = infrastructure_params[infrastructure_id]["expected_cnodes"]
    return n_actions


def construct_model(trf: dict, fcnn_params: dict, n_actions: int):
    return AttentionQFunction_D_DQN(n_actions=n_actions, **trf, **fcnn_params)


def construct_explorer(explorer_params: dict, n_actions: int):

    if explorer_params["choice"] == "decay_epsilon_greedy":
        return pfrl.explorers.LinearDecayEpsilonGreedy(random_action_func=lambda : np.random.choice([i for i in range(n_actions)]), **explorer_params["decay_epsilon_greedy"])
    
    if explorer_params["choice"] == "constant_epsilon_greedy":
        return pfrl.explorers.ConstantEpsilonGreedy(random_action_func=lambda : np.random.choice([i for i in range(n_actions)]), **explorer_params["constant_epsilon_greedy"])


def construct_replay_buffer(buffer_params: dict):
    if buffer_params["choice"] == "prioritized":
        return pfrl.replay_buffers.PrioritizedReplayBuffer(**buffer_params["prioritized"])
    
    if buffer_params["choice"] == "normal":
        return pfrl.replay_buffers.ReplayBuffer(**buffer_params["normal"])


def custom_batch_states(states, device, phi):
    sequences = []
    for state in states:
        sequences.append( state )
    #------------------------------------------
    sequences = torch.tensor(sequences, dtype=torch.float32, device=device)
    return sequences

def construct_optimizer_and_agent(model, replay_buffer, explorer, optimizer_params: dict, algo_params: dict):
    opt = None
    if optimizer_params["choice"] == "lion":
        opt = Lion(model.parameters(), **optimizer_params["lion"])
    
    if optimizer_params["choice"] == "adam":
        opt = Adam(model.parameters(), **optimizer_params["adam"])
    
    phi = lambda x: x
    if algo_params["choice"] == "dqn":
        return pfrl.agents.DQN(q_function=model, optimizer=opt, replay_buffer=replay_buffer, explorer=explorer, phi=phi, batch_states=custom_batch_states, **algo_params["dqn_ddqn"])

    if algo_params["choice"] == "ddqn":
        return pfrl.agents.DoubleDQN(q_function=model, optimizer=opt, replay_buffer=replay_buffer, explorer=explorer, phi=phi, batch_states=custom_batch_states, **algo_params["dqn_ddqn"])


#======================================================================
# Nodes to make agent and envionments interact
#======================================================================

def run_one_episode(episode, perf_basename, agent, focus, environment, infrastructure_id, infra_seed, nspr_seed):
    observation = environment.reset(infrastructure_id, infra_seed, nspr_seed)
    iteration = 1
    while True:
        observation, f_idx = focus.apply_focus(observation)
        action = agent.act(observation)
        observation, reward, done, info = environment.step(action)
        reset = done
        fobservation = focus.extract_focus(observation, f_idx)
        try:
            agent.observe(fobservation, reward, done, reset)
        except:
            with open(f"aaa_crashed_{perf_basename}_eps_{episode}_iter_{iteration}", 'w'):
                pass
            import time ; time.sleep(30)
        if done:
            break
        iteration += 1

def agent_and_environments_interaction(agent, focus, t_environment, e_environment, infrastructure_id, infra_train_seed, nspr_train_seed, infra_eval_seed, nspr_eval_seed, loop: dict, perf_file):
    n_training_episodes = loop["n_train_episodes"]
    evaluate_interval = loop["evaluate_interval"]
    performance_record = []

    incrementer = 0
    perf_file_without_extension = perf_file.split('.')[0]

    for i in range(n_training_episodes):
        run_one_episode(i, perf_file_without_extension, agent, focus, t_environment, infrastructure_id, infra_train_seed + incrementer, nspr_train_seed + incrementer)
        incrementer += 1
        
        if i % evaluate_interval == 0:
            with agent.eval_mode():
                run_one_episode(i, perf_file_without_extension, agent, focus, e_environment, infrastructure_id, infra_eval_seed, nspr_eval_seed)
                performance_record.append( e_environment.statistics() )
        if i % 500 == 0:
            print(f"--> episode {i}")
        
        if i % 5 == 0:
            with open(perf_file, 'a+') as f:
                numbers_to_add = "\n".join([str(v) for v in performance_record])
                f.write(numbers_to_add)
            performance_record = []