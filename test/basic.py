from pupperv3_mjx import BarkourEnv, domain_randomize, progress
from brax import envs
from datetime import datetime
import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

envs.register_environment("barkour", BarkourEnv)
env_name = "barkour"

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 40, 0

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 128, 128)
)
train_fn = functools.partial(
    ppo.train,
    num_timesteps=100_000_000,
    num_evals=10,
    reward_scaling=1,
    episode_length=10,  # 1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=1,  # 4,
    discounting=0.97,
    learning_rate=3.0e-4,
    entropy_cost=1e-2,
    num_envs=1,
    batch_size=8,  # 256,
    network_factory=make_networks_factory,
    randomization_fn=domain_randomize,
    seed=0,
)

# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=lambda num_steps, metrics: progress(
        num_steps=num_steps,
        metrics=metrics,
        times=times,
        x_data=x_data,
        y_data=y_data,
        ydataerr=ydataerr,
        train_fn=train_fn,
        max_y=max_y,
        min_y=min_y,
    ),
    eval_env=eval_env,
)
