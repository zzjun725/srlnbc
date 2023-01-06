import ray.tune
import ray.tune.utils.log
from ray.rllib.agents.callbacks import MultiCallbacks
from ray.rllib.agents.ppo.ppo import PPOTrainer
from srlnbc.agents.callbacks.metadrive import DrivingCallbacks
from srlnbc.env.register import register_metadrive
from srlnbc.utils.tune import create_progress

register_metadrive()

stop_criteria = {
    'timesteps_total': int(2e7),
}

agent_config = {
    # Env config
    'env': 'metadrive',

    # Worker config
    'framework': 'tf2',
    'eager_tracing': True,
    'num_workers': 40,
    'num_envs_per_worker': 1,
    'num_gpus': 0,

    # PPO config
    'rollout_fragment_length': 1000,
    'train_batch_size': 40000,
    'sgd_minibatch_size': 40000,
    'num_sgd_iter': 30,
    'lr': 3e-4,
    'lambda': 0.95,
    'gamma': 0.99,
    'no_done_at_end': False,

    # Model config
    'model': {
        'fcnet_hiddens': [256, 256],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,
        'free_log_std': True,
    },
    'callbacks': MultiCallbacks([
        DrivingCallbacks,
    ]),
    "seed": 1,
}

ray.tune.run(
    PPOTrainer,
    checkpoint_freq=50,
    keep_checkpoints_num=3,
    checkpoint_at_end=True,
    stop=stop_criteria,
    config=agent_config,
    verbose=ray.tune.utils.log.Verbosity.V1_EXPERIMENT,
    progress_reporter=create_progress(),
    log_to_file=True,
    max_concurrent_trials=1
)
