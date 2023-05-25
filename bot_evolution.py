import random
from datetime import datetime
from bot_trainer import BotTrainer
from hex_bot import HexBot
from hex_game import HexGame
import os
import torch
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool


class BotEvolution:
    """ Performs multiple generations of training, playing off models against one another. """
    # TODO(cw/cd): add automatic storing of metadata with the models

    def __init__(self, rootfolder=".", hex_size=8, generations=3, bots_per_generation=3, start_bots=None, start_opponent_policies=None):
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

    def play1v1(self, bot1: HexBot, bot2: HexBot, nr_games: int = 1000):
        """ 
        Play two bots off against one another. 

        Return the ratio of games won by bot1.
        """
        score1 = 0
        for game_idx in range(nr_games):
            game = HexGame(
                hex_size=self.hex_size, start_color=HexGame.RED if game_idx % 2 == 0 else HexGame.BLUE,
                opponent_policy=bot2.play_policy
            )
            bot1_wins = game.play_out(bot1.play_policy)
            if bot1_wins:
                score1 += 1
        return score1 / nr_games

    def save_bot(self, bot: HexBot, modelname, filename, metadata: Optional[Dict] = None):
        """ Save bot state and performance metadata. """
        # TODO(c): add some metadata to a 'global' data file in
        #          the root directory of the run
        data = {
            "name": modelname,
            "meta": metadata,
            "bot": bot
        }

        torch.save(data, filename)

    def load_bot(self, filename):
        data = torch.load(filename)
        return data["bot"]

    def init_gen_worker_(self, gen_, opponent_pool_, botdata_folder_):
        global gen
        global opponent_pool
        global botdata_folder

        gen = gen_
        opponent_pool = opponent_pool_
        botdata_folder = botdata_folder_

    def train_async(self, idx_bot):
        idx, bot = idx_bot

        print(f"Training bot {idx+1}/{self.bots_per_generation}.")
        # TODO(cw/cd): introduce further hyperparameters as params to BotTrainer
        trainer = BotTrainer(bot, game_size=self.hex_size,
                             opponent_pool=opponent_pool,
                             sampling_updates=1,
                             episodes_per_sampling=1
                             )
        # Train bot - output metadata (a Dict)
        # metadata should in particular include
        # an overall "score"
        metadata = trainer.train()
        botname = f"gen{gen+1}bot{idx+1}"
        filename = f"{botdata_folder}/{botname}"

        self.save_bot(bot, botname, filename, metadata)

        return metadata["score"]

    def evolve(self, subfolder=None):
        if subfolder is None:
            date_prefix = datetime.now().date().strftime("%Y%m%d")
            # generate random postfix
            hash_postfix = "%x" % random.getrandbits(32)
            subfolder = date_prefix + "_" + hash_postfix

        folder = f"{self.rootfolder}/{subfolder}"
        os.mkdir(folder)
        botdata_folder = f"{folder}/botdata"
        os.mkdir(botdata_folder)
        report_file = f"{folder}/report.md"
        data_file = f"{folder}/data"

        opponent_pool = []  # start playing against random policy
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")

            scores = [0. for _ in range(self.bots_per_generation)]

            idx_bot = enumerate(self.bots)

            with Pool(processes=self.bots_per_generation, initializer=self.init_gen_worker_, initargs=(gen, opponent_pool, botdata_folder)) as pool:
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

            # TODO(cw/cd): output metadata to file here

            # Update opponent pool:
            #  1. half the weight of existing opponents
            #  2. add the bots of this round with weight 1
            opponent_pool = [(weight/2, policy)
                             for weight, policy in opponent_pool]
            opponent_pool += [(1., bot.play_policy) for bot in self.bots]

            # Derive next generation of bots from this generation,
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
    bot_evo.evolve()


if __name__ == "__main__":
    main()
