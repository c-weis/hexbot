from bot_trainer import BotTrainer
from hex_bot import HexBotBrain
from hex_game import HexGame
import os
import torch
from typing import Dict, List, Tuple, Optional
import random
import time
from multiprocessing import Pool

class BotTrainerTemp: 
    def __init__(self, id):
        self.id = id
        self.trainee = "nn.Module"

    def train(self):
        print(f"training bot {self.id} ... wow it takes so long")
        time.sleep(2)
        return { "id" : self.id, "score" : random.randint(0,100) }


class BotEvolution:
    """ Performs multiple generations of training, playing off models against one another. """
    # TODO(cw/cd): add automatic storing of metadata with the models

    def __init__(self, hex_size=8, generations=10, bots_per_generation=5, start_bots=None, start_opponent_policies=None):
        self.hex_size = hex_size
        self.generations = generations
        self.bots_per_generation = bots_per_generation
        self.bots = [i for i in range(self.bots_per_generation)]
        self.folder="test_parallel"

    def train_async(self, idx_bot : Tuple[int,int]):
            # print(id(self))
            idx, bot = idx_bot
            trainer = BotTrainerTemp(id=idx) 
            metadata = trainer.train()
            gen = 1
            folder = self.folder
            botname = f"gen{gen+1}bot{idx+1}"
            botdata_folder = f"{folder}/botdata"
            os.makedirs(botdata_folder, exist_ok=True)
            filename = f"{botdata_folder}/gen{gen+1}bot{idx+1}"
            return metadata["score"]


    def evolve_parallel(self):
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")

            scores = [0. for _ in range(self.bots_per_generation)]
            print("start")
            with Pool(processes = self.bots_per_generation) as pool:
                scores = pool.map(self.train_async, enumerate(self.bots))

                # for idx, bot in enumerate(self.bots):
                #     metadata_result = pool.apply_async(self.train_async, idx, bot)
                #     scores[idx] = metadata_result.get(timeout=10)

            # Sort bots by score
            sorted_scores = sorted(
                enumerate(scores), key=lambda idx_score: idx_score[1], reverse=True)
            sorted_bots = [(score, idx, self.bots[idx])
                           for idx, score in sorted_scores]

            # Output scores to terminal
            print("Bot scores in descending order:")
            for score, idx, _ in sorted_bots:
                print(f"{score}, bot {idx+1}")

            # currently: cycle through top third
            nr_bots_kept = self.bots_per_generation//3 
            for bot_idx in range(self.bots_per_generation):
                self.bots[bot_idx] = sorted_bots[bot_idx % nr_bots_kept][2]

        print("Evolution cycle complete.")
        # TODO(CW/CD): add more output here
        # output best bot weights to a specific file (or create link) 
        # output metadata about evolution process to file

    def evolve_non_parallel(self):
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")

            scores = [0. for _ in range(self.bots_per_generation)]
            for idx, bot in enumerate(self.bots):
                print(f"Training bot {idx+1}/{self.bots_per_generation}.")
                trainer = BotTrainerTemp(id=idx) 
                metadata = trainer.train()

                botname = f"gen{gen+1}bot{idx+1}"
                botdata_folder = f"{self.folder}/botdata"
                os.makedirs(botdata_folder, exist_ok=True)
                filename = f"{botdata_folder}/gen{gen+1}bot{idx+1}"

                scores[idx] = metadata["score"]

            # Sort bots by score
            sorted_scores = sorted(
                enumerate(scores), key=lambda idx_score: idx_score[1], reverse=True)
            sorted_bots = [(score, idx, self.bots[idx])
                           for idx, score in sorted_scores]

            # Output scores to terminal
            print("Bot scores in descending order:")
            for score, idx, _ in sorted_bots:
                print(f"{score}, bot {idx+1}")

            # currently: cycle through top third
            nr_bots_kept = self.bots_per_generation//3 
            for bot_idx in range(self.bots_per_generation):
                self.bots[bot_idx] = sorted_bots[bot_idx % nr_bots_kept][2]

        print("Evolution cycle complete.")
        # TODO(CW/CD): add more output here
        # output best bot weights to a specific file (or create link) 
        # output metadata about evolution process to file


def main():
    bot_evo = BotEvolution()
    bot_evo.evolve_non_parallel()


if __name__ == "__main__":
    main()
