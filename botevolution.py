from os import makedirs, mkdir
import random
from copy import deepcopy
from datetime import datetime
import torch
from typing import Callable, Dict, List, Tuple, Optional
from multiprocessing import Pool
from hexbot import HexBot
from hexgame import HexGame
from bottrainer import BotTrainer


class BotEvolution:
    """ Environment to perform multiple generations of training. """

    def __init__(self,
                 game_size: int = 5, generations: int = 15,
                 bots_per_generation: int = 15,
                 rootfolder: str = "./output/",
                 start_bots: List[HexBot] = None,
                 start_opponent_policies: List[Tuple[float, Callable]] = None,
                 monitoring: bool = False,
                 trainer_args: Dict = None):
        self.rootfolder = rootfolder

        self.game_size = game_size

        self.generations = generations
        self.bots_per_generation = bots_per_generation
        if start_bots is None:
            self.bots = [
                HexBot(game_size=self.game_size)   # adjust here
                for _ in range(self.bots_per_generation)
            ]
        else:
            self.bots = start_bots
        self.opponent_policies = start_opponent_policies

        self.human_monitoring_enabled = monitoring

        self.trainer_args = trainer_args

    def play1v1(self, bot1: HexBot, bot2: HexBot, nr_games: int = 1000, render: bool = False) -> float:
        """ 
        Play two bots off against one another. 

        Keywords:
        bot1, bot2: two HexBots
        nr_games: number of games to play
        render: if True, displays games

        Return the ratio of games won by bot1.
        """
        score1 = 0
        for game_idx in range(nr_games):
            game = HexGame(
                size=self.game_size,
                start_color=HexGame.RED if game_idx % 2 == 0 else HexGame.BLUE,
                render_enabled=render,
                opponent_policy=bot2.play_policy
            )
            bot1_wins = game.play_out(bot1.play_policy)
            if bot1_wins:
                score1 += 1
        return score1 / nr_games

    def save_bot(self, bot: HexBot, modelname: str, filename: str, metadata: Optional[Dict] = None):
        """ 
        Save bot state and performance metadata. 

        """
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

        Keywords:
        idx_bot: tuple containing bot index and bot (nn.Module)

        Return performance score (=win rate) between 0 and 1.
        """
        idx, bot = idx_bot

        print(f"Training bot {idx+1}/{self.bots_per_generation}.")

        if self.trainer_args is None:
            trainer = BotTrainer(bot, game_size=self.game_size,
                                 opponent_pool=opponent_pool,
                                 sampling_updates=50,
                                 episodes_per_sampling=5
                                 )
        else:
            trainer = BotTrainer(bot, game_size=self.game_size,
                                 opponent_pool=opponent_pool,
                                 **self.trainer_args)
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

        Keywords:
        subfolder: optional string setting a subfolder for output \
            if none is given, a subfolder is generated from the date \
            and a random hash string
        """
        if subfolder is None:
            date_prefix = datetime.now().strftime("%y%m%d%H%M")
            # generate random postfix
            hash_postfix = "%x" % random.getrandbits(16)
            subfolder = date_prefix + "_" + hash_postfix

        makedirs(self.rootfolder, exist_ok=True)
        folder = f"{self.rootfolder}/{subfolder}"
        mkdir(folder)
        botdata_folder = f"{folder}/botdata"
        mkdir(botdata_folder)

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
                    bot1=self.bots[0], bot2=self.bots[1], nr_games=1, render=True)

        print("Evolution cycle complete.")


def make_them_play(bot1_filename: str, bot2_filename: str, nr_games: int, rootfolder: str = "./output/"):
    """ 
    Display a given number of games between bot1 and bot2. 

    Keywords:
    bot1_filename, bot2_filename: file strings relative to `rootfolder`
    nr_games: number of games to play
    rootfolder: specifies root folder - defaults to "./output/"
    """
    bot_evo_arena = BotEvolution()
    bot1data = torch.load(rootfolder + bot1_filename)
    bot1 = bot1data["bot"]
    bot2data = torch.load(rootfolder + bot2_filename)
    bot2 = bot2data["bot"]
    bot_evo_arena.play1v1(bot1, bot2, nr_games, render=True)


def test():
    """ Run an evolution cycle with test parameters. """
    game_size = 5
    trainer_args = {
        "parallel_workers": 2,
        "sampling_steps": 16,
        "sampling_updates": 3
    }
    bot_evo = BotEvolution(game_size,
                           generations=3,
                           bots_per_generation=3,
                           rootfolder="./test/",
                           monitoring=True,
                           trainer_args=trainer_args)
    bot_evo.evolve()


if __name__ == "__main__":
    test()
