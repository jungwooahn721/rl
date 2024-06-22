import os
import argparse

import gymnasium as gym
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

import aai4160.env_configs
from aai4160 import envs
from aai4160.envs import register_envs
from aai4160.agents.model_based_agent import ModelBasedAgent
from aai4160.infrastructure import pytorch_util as ptu
from aai4160.infrastructure import utils
from aai4160.infrastructure.logger import Logger
from aai4160.infrastructure.replay_buffer import ReplayBuffer

from scripting_utils import make_logger, make_config


register_envs()


def run_training_loop(
    config: dict, logger: Logger, args: argparse.Namespace,
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our MPC implementation only supports continuous action spaces."

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 2

    # initialize agent
    mb_agent = ModelBasedAgent(
        env,
        **config["agent_kwargs"],
    )
    actor_agent = mb_agent

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    total_envsteps = 0
    random_policy = utils.RandomPolicy(env)

    for itr in range(config["num_iters"]):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        if itr == 0:
            # TODO(student): collect at least config["initial_batch_size"] transitions with a random policy
            # HINT: Use `random_policy` and `utils.sample_trajectories`
            ### STUDENT CODE BEGIN HERE ###
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env,
                policy=random_policy,
                min_timesteps_per_batch=config["initial_batch_size"],
                max_length=ep_len,
            )
            ### STUDENT CODE END HERE ###
        else:
            # TODO(student): collect at least config["batch_size"] transitions with our `actor_agent`
            ### STUDENT CODE BEGIN HERE ###
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env,
                policy=actor_agent,
                min_timesteps_per_batch=config["batch_size"],
                max_length=ep_len,
            )
            ### STUDENT CODE END HERE ###

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        for traj in trajs:
            replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # update agent's statistics with the entire replay buffer
        mb_agent.update_statistics(
            obs=replay_buffer.observations[: len(replay_buffer)],
            acs=replay_buffer.actions[: len(replay_buffer)],
            next_obs=replay_buffer.next_observations[: len(replay_buffer)],
        )

        # train agent
        print("Training model...")
        all_losses = []
        for _ in tqdm.trange(
            config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            step_losses = []
            # TODO(student): train the dynamics models
            # HINT: train each dynamics model in the ensemble with a *different* batch of transitions using mb_agent.update()
            # You may use for loop with size of `mb_agent.ensemble_size`
            # Use `replay_buffer.sample` with config["train_batch_size"].
            
            '''
                def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
                    """
                    Update self.dynamics_models[i] using the given batch of data.

                    Args:
                        i: index of the dynamics model to update
                        obs: (batch_size, ob_dim)
                        acs: (batch_size, ac_dim)
                        next_obs: (batch_size, ob_dim)
                    """
                    obs = ptu.from_numpy(obs)
                    acs = ptu.from_numpy(acs)
                    next_obs = ptu.from_numpy(next_obs)

                    obs_acs = torch.concatenate([obs, acs], axis = -1)
                    obs_acs_normalized = (obs_acs - self.obs_acs_mean) / self.obs_acs_std

                    obs_delta = next_obs - obs
                    obs_delta_normalized = (obs_delta - self.obs_delta_mean) / self.obs_delta_std

                    # TODO(student): update self.dynamics_models[i] using the given batch of data
                    # HINT: use self.dynamics_models[i] to get the normalized delta prediction for next observation.
                    # Note that the model recieves normalized observation-action for its input.
                    # Optimize the model with squared loss.
                    ### STUDENT CODE BEGIN HERE ###
                    obs_delta_normalized_hat = self.dynamics_models[i](obs_acs_normalized)
                    loss = self.loss_fn(obs_delta_normalized_hat, obs_delta_normalized)
                
                    ### STUDENT CODE END HERE ###

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    return ptu.to_numpy(loss)
            '''
            ### STUDENT CODE BEGIN HERE ###
            # step_losses = mb_agent.update(
            #     replay_buffer.sample(config["train_batch_size"])
            # ) -> ModelBasedAgent.update() missing 3 required positional arguments: 'obs', 'acs', and 'next_obs'   
            sample = replay_buffer.sample(config["train_batch_size"])
            # step_losses = mb_agent.update(sample['observations'], sample['actions'], sample['next_observations'])
            step_losses = [mb_agent.update(i, sample['observations'], sample['actions'], sample['next_observations']) for i in range(mb_agent.ensemble_size)]
            ### STUDENT CODE END HERE ###

            all_losses.append(np.mean(step_losses))

        # on iteration 0, plot the full learning curve
        if itr == 0:
            plt.plot(all_losses)
            plt.title("Iteration 0: Dynamics Model Training Loss")
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.savefig(os.path.join(logger._log_dir, "itr_0_loss_curve.png"))

        # log the average loss
        loss = np.mean(all_losses)
        logger.log_scalar(loss, "dynamics_loss", itr)

        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue

        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return (mean): {np.mean(returns)}")
        print(f"Average eval return (std) : {np.std(returns)}")

        if len(returns) > 1:
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    actor_agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    itr,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    config = make_config(args.config_file)

    logger = make_logger(config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
