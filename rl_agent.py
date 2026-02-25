from datetime import datetime

import numpy as np
from collections import deque
import random
import os

import tensorflow as tf

from game import Game

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

class RLAgent:
    def __init__(self, game: Game, buffer_size=10000, batch_size=64):
        self.game = game
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.last_obs = None
        self.last_action = None
        self.runs = []

        self.model_path = "rl_agent_model.keras"
        self._configure_gpu_memory_growth()
        self.model = self.build_or_load_model()

    @staticmethod
    def _configure_gpu_memory_growth():
        """Avoid TensorFlow grabbing all GPU memory up-front."""
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth has to be configured before GPU initialization.
                pass

    def build_or_load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            return tf.keras.models.load_model(self.model_path)
        else:
            return self.build_model()

    def build_model(self):
        # Simple feedforward network
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.game.cfg.rays.n_rays + 1,)),  # ray distances + speed
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')  # throttle, brake, steer
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def get_inputs(self, observation):
        """
        Given the current observation, return a dict with keys:
        'throttle', 'brake', 'steer'
        """
        # Epsilon-greedy exploration
        epsilon = 0.1
        if np.random.rand() < epsilon:
            action = {
                "throttle": np.random.uniform(0.9, 1),
                "brake": np.random.uniform(0, 0.3),
                "steer": np.random.uniform(-1, 1),
            }
        else:
            prediction = self.model.predict(np.array([observation["ray_distances"] + [observation["speed"]]]), verbose=0)
            action = {
                "throttle": clamp(prediction[0][0], 0, 1),
                "brake": clamp(prediction[0][1], 0, 1),
                "steer": clamp(prediction[0][2], -1, 1),
            }

        # Prevent throttle if speed is negative (reverse)
        if observation["speed"] < 0:
            action["brake"] = 0

        self.last_obs = observation
        self.last_action = action
        return action
    
    def store_run(self, reward, laptime, distance):
        self.runs.append({
            "timestamp": datetime.now().isoformat(),
            "reward": reward,
            "laptime": laptime,
            "distance": distance,
        })
        print(f"Run {len(self.runs)}: reward={reward:.2f}, laptime={laptime:.2f}s, distance={distance:.1f}px")

    def feed_back(self, events, observation):
        """
        Receive feedback after each step.
        Use this to store transitions, update your model, etc.
        """
        # Define reward
        reward = observation["travelled_distance"]
        if events.get("crash"):
            reward -= 100  # Penalty for crashing

        # Penalize or zero reward for backwards driving
        if observation["speed"] < 0:
            reward += -50  # stronger penalty for reverse
        if observation["speed"] == 0:
            reward += -10  # no reward for being stationary
        if observation["speed"] > 0:
            reward += 10  # no reward for being stationary

        done = events.get("crash") or events.get("quit")
        # Store transition
        if self.last_obs is not None and self.last_action is not None:
            state_input = np.asarray(self.last_obs["ray_distances"] + [self.last_obs["speed"]], dtype=np.float32)
            self.replay_buffer.append(
                (state_input, self.last_action, reward)
            )

        # Train model if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self.train_model()

        if done:
            self.store_run(reward, observation["lap_time"], observation["travelled_distance"])
            if events.get("quit"):
                self.save_model()
                self.save_runs()
            self.game.reset()
            self.last_obs = None
            self.last_action = None

    def save_model(self):
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def save_runs(self):
        if not os.path.exists("rl_agent_runs.csv"):
            with open("rl_agent_runs.csv", "w") as f:
                f.write("timestamp,reward,laptime,distance\n")
        with open("rl_agent_runs.csv", "a") as f:
            for run in self.runs:
                f.write(f"{run['timestamp']},{run['reward']},{run['laptime']},{run['distance']}\n")
        print(f"Runs saved to rl_agent_runs.csv")

    def train_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = []
        targets = []

        for state_input, action, reward in batch:
            # Target is the action taken, adjusted by reward
            target = np.array([
                clamp(action["throttle"] + reward * 0.01, 0, 1),
                clamp(action["brake"] + reward * 0.01, 0, 1),
                clamp(action["steer"] + reward * 0.01, -1, 1),
            ])
            states.append(state_input)
            targets.append(target)

        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, epochs=1, verbose=0)
