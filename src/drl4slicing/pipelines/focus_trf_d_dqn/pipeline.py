"""
This is a boilerplate pipeline 'focus_trf_d_dqn'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import construct_infrastructures_managers, construct_nsprs_generators, construct_environments, construct_focus_module, determine_agent_action_space_size, construct_model, construct_replay_buffer, construct_explorer, agent_and_environments_interaction, construct_optimizer_and_agent

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        #======================================================================
        # Nodes to construct environments
        #======================================================================
        node(
            func=construct_infrastructures_managers,
            inputs=["params:infrastructure"],
            outputs=["train_infrastructure_manager_ftdd", "eval_infrastructure_manager_ftdd"],
            name="infras_man_node_ftdd"
        ),
        node(
            func=construct_nsprs_generators,
            inputs="params:nspr",
            outputs=["train_nsprs_generator_ftdd", "eval_nsprs_generator_ftdd"],
            name="nsprs_gen_node_ftdd"
        ),
        node(
            func=construct_environments,
            inputs=["train_infrastructure_manager_ftdd",
                    "train_nsprs_generator_ftdd",
                    "eval_infrastructure_manager_ftdd",
                    "eval_nsprs_generator_ftdd"],
            outputs=["train_environment_ftdd", "eval_environment_ftdd"],
            name="envs_node_ftdd"
        ),
        #======================================================================
        # Nodes to construct agent
        #======================================================================
        node(
            func=construct_focus_module,
            inputs=["params:ftdd.focus"],
            outputs="focus_ftdd",
            name="focus_node_ftdd"
        ),
        node(
            func=determine_agent_action_space_size,
            inputs=["params:ftdd.focus",
                    "params:infrastructure"],
            outputs="action_space_size_ftdd",
            name="action_space_node_ftdd"
        ),
        node(
            func=construct_model,
            inputs=["params:ftdd.trf",
                    "params:ftdd.fcnn",
                    "action_space_size_ftdd"],
            outputs="model_ftdd",
            name="model_fcnn_node_ftdd"
        ),
        node(
            func=construct_explorer,
            inputs=["params:ftdd.explorer",
                    "action_space_size_ftdd"],
            outputs="explorer_ftdd",
            name="explorer_node_ftdd"
        ),
        node(
            func=construct_replay_buffer,
            inputs=["params:ftdd.replay_buffer"],
            outputs="replay_buffer_ftdd",
            name="buffer_node_ftdd"
        ),
        node(
            func=construct_optimizer_and_agent,
            inputs=["model_ftdd",
                    "replay_buffer_ftdd",
                    "explorer_ftdd",
                    "params:ftdd.optimizer",
                    "params:ftdd.algorithm"],
            outputs="agent_ftdd",
            name="agent_node_ftdd"
        ),
        #======================================================================
        # Nodes to make agent and envionments interact
        #======================================================================
        node(
            func=agent_and_environments_interaction,
            inputs=["agent_ftdd",
                    "focus_ftdd",
                    "train_environment_ftdd",
                    "eval_environment_ftdd",
                    "params:infrastructure.choice",
                    "params:infrastructure.train_start_seed",
                    "params:nspr.train_start_seed",
                    "params:infrastructure.eval_seed",
                    "params:nspr.eval_seed",
                    "params:ftdd.drl_loop",
                    "params:ftdd.save_perf_in"],
            outputs=None,
            name="interaction_node_ftdd"
        ),
    ])