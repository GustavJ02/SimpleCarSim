from game import Game
from config import GameConfig
from rl_agent import RLAgent
import random

class RandomPolicy:
    def __init__(self):
        pass

    def get_inputs(self, observation):
        return {
            "throttle": random.uniform(0, 1),
            "brake": random.uniform(0, 0.3),
            "steer": random.uniform(-1, 1),
        }
    
    def feed_back(self, events, observation):
        evs = ",".join(f"{k}={v}" for k, v in events.items())
        print("Events:", evs, end="\r", flush=True)


def main():
    game = Game(GameConfig())
    driver = RLAgent(game)
    game.run(input_obj=driver)


if __name__ == "__main__":
    main()