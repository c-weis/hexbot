import os
import random
from copy import deepcopy
from datetime import datetime
import torch
from typing import Callable, Dict, List, Tuple, Optional
from multiprocessing import Pool
from hex_bot import HexBot
from hex_game import HexGame
from bot_trainer import BotTrainer


class BotEvolution:
    """ 
    Environment to perform multiple generations of training.
    """

    def __init__(self,
                 rootfolder: str = "./bot_evolution_output/",
                 hex_size: int = 5, generations: int = 15,
                 bots_per_generation: int = 15,
                 start_bots: List[HexBot] = None,
                 start_opponent_policies: List[Tuple[float, Callable]] = None,
                 monitoring: bool = False):
        self.rootfolder = rootfolder

        self.hex_size = hex_size

        self.generations = generations
        self.bots_per_generation = bots_per_generation
        if start_bots is None:
            self.bots = [
                HexBot(hex_size=self.hex_size)   # adjust here
                for _ in range(self.bots_per_generation)
            ]
        else:
            self.bots = start_bots
        self.opponent_policies = start_opponent_policies

        self.human_monitoring_enabled = monitoring

    def play1v1(self, bot1: HexBot, bot2: HexBot, nr_games: int = 1000, render: str = "none") -> float:
        """ 
        Play two bots off against one another. 

        Return the ratio of games won by bot1.
        """
        score1 = 0
        for game_idx in range(nr_games):
            game = HexGame(
                size=self.hex_size,
                start_color=HexGame.RED if game_idx % 2 == 0 else HexGame.BLUE,
                render_mode=render,
                opponent_policy=bot2.play_policy
            )
            bot1_wins = game.play_out(bot1.play_policy)
            if bot1_wins:
                score1 += 1
        return score1 / nr_games

    def save_bot(self, bot: HexBot, modelname: str, filename: str, metadata: Optional[Dict] = None):
        """ Save bot state and performance metadata. """
        data = {
            "name": modelname,
            "meta": metadata,
            "bot": bot
        }

        torch.save(data, filename)

    def load_bot(self, filename: str):
        """ Load bot from file. """
        data = torch.load(filename)
        return data["bot"]

    def init_gen_worker_(self, gen_: int, opponent_pool_: List[Tuple[float, Callable]], botdata_folder_: str):
        """ [Multiprocessing] Initialise global variables for worker. """
        global gen
        global opponent_pool
        global botdata_folder

        gen = gen_
        opponent_pool = opponent_pool_
        botdata_folder = botdata_folder_

    def train_async(self, idx_bot: Tuple[int, HexBot]) -> float:
        """ 
        Perform training on HexBot `bot` with index `idx`. 

        Return performance score (=win rate) between 0 and 1.
        """
        idx, bot = idx_bot

        print(f"Training bot {idx+1}/{self.bots_per_generation}.")
        trainer = BotTrainer(bot, game_size=self.hex_size,
                             opponent_pool=opponent_pool,
                             sampling_updates=50,
                             episodes_per_sampling=5
                             )
        # Train bot - output metadata (a Dict)
        # metadata should in particular include
        # an overall "score"
        metadata = trainer.train()
        botname = f"gen{gen+1}bot{idx+1}"
        filename = f"{botdata_folder}/{botname}"

        self.save_bot(bot, botname, filename, metadata)

        return metadata["score"]

    def evolve(self, subfolder: str = None):
        """ 
        Run multiple generations of training.
        """
        if subfolder is None:
            date_prefix = datetime.now().strftime("%y%m%d%H%M")
            # generate random postfix
            hash_postfix = "%x" % random.getrandbits(16)
            subfolder = date_prefix + "_" + hash_postfix

        folder = f"{self.rootfolder}/{subfolder}"
        os.mkdir(folder)
        botdata_folder = f"{folder}/botdata"
        os.mkdir(botdata_folder)

        opponent_pool = []  # start playing against random policy
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")

            scores = [0. for _ in range(self.bots_per_generation)]

            idx_bot = enumerate(self.bots)

            with Pool(processes=self.bots_per_generation,
                      initializer=self.init_gen_worker_,
                      initargs=(gen, opponent_pool, botdata_folder)) as pool:
                scores = pool.map(self.train_async, idx_bot)

            # Sort bots by score
            sorted_scores = sorted(
                enumerate(scores), key=lambda idx_score: idx_score[1], reverse=True)
            sorted_bots = [(score, idx, self.bots[idx])
                           for idx, score in sorted_scores]

            # Output scores to terminal
            print("Bot scores in descending order:")
            for score, idx, _ in sorted_bots:
                print(f"{score}, bot {idx+1}")

            # Update opponent pool:
            #  1. decrease the weight of existing opponents
            #  2. add the bots of this round with weight 1
            opponent_pool = [(weight/1.02, policy)
                             for weight, policy in opponent_pool]
            opponent_pool += [(1., bot.play_policy) for bot in self.bots]

            # Derive next generation of bots from this generation:
            # discard the worst, double the best
            nr_bots_kept = self.bots_per_generation-1
            for bot_idx in range(self.bots_per_generation):
                self.bots[bot_idx] = deepcopy(
                    sorted_bots[bot_idx % nr_bots_kept][2])

            if self.human_monitoring_enabled:
                self.play1v1(
                    bot1=self.bots[0], bot2=self.bots[1], nr_games=1, render="human")

        print("Evolution cycle complete.")


def make_them_play(bot1_filename: str, bot2_filename: str, nr_games: int):
    """ Display a given number of games between bot1 and bot2. """
    bot_evo_arena = BotEvolution()
    bot1data = torch.load("./bot_evolution_output/" + bot1_filename)
    bot1 = bot1data["bot"]
    bot2data = torch.load("./bot_evolution_output/" + bot2_filename)
    bot2 = bot2data["bot"]
    bot_evo_arena.play1v1(bot1, bot2, nr_games, "human")


def test():
    """ Run an evolution cycle with default parameters. """
    bot_evo = BotEvolution(monitoring=True)
    bot_evo.evolve()


if __name__ == "__main__":
    test()
