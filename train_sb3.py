from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from env.car_env import CarEnv


def main():
    # Vectorized env speeds up training and stabilizes learning
    env = make_vec_env(lambda: CarEnv(render_mode="none"), n_envs=8)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./runs/",
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=200_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
    )

    model.learn(total_timesteps=300_000)
    model.save("sac_car")

    env.close()


if __name__ == "__main__":
    main()
