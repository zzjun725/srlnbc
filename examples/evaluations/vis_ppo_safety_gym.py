import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from srlnbc.env.register import register_safety_gym

# Load the trained model from the checkpoint
CHECKPOINT_PATH = '/home/zhijunz/ray_results/' + \
                  'PPOTrainer_2024-02-15_12-11-36/PPOTrainer_point_goal_469e5_00000_0_2024-02-15_12-11-36/' + \
                  'checkpoint_000500/checkpoint-500'

# Register your custom environments
register_safety_gym()


# Configuration for the agent, make sure it matches the training configuration
agent_config = {
    'env': 'point_goal',
    # Make sure the rest of the configuration matches your training setup
    'framework': 'tf2',
    'num_workers': 0,  # For visualization, it's often set to 0
    'num_gpus': 0,
    'model': {
        'fcnet_hiddens': [256, 256],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,
        'free_log_std': True,
    },
    # Include other necessary configurations that match your training setup
}

# Initialize the Trainer with the environment and the same configuration used during training
agent = PPOTrainer(config=agent_config, env='point_goal')

# Restore the agent from the checkpoint
agent.restore(CHECKPOINT_PATH)

# Now that the environment is correctly registered, you can directly instantiate it
env = agent.workers.local_worker().env

state = env.reset()
done = False
cumulative_reward = 0

while not done:
    action = agent.compute_single_action(state)
    state, reward, done, info = env.step(action)
    cumulative_reward += reward
    # If your environment supports rendering, you can add it here
    env.render()

print(f'Cumulative reward: {cumulative_reward}')

# Cleanup
env.close()
ray.shutdown()