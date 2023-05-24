from bot_trainer import BotTrainer
from hex_bot import HexBotBrain
from hex_game import HexGame
import os


class BotEvolution:
    """ Performs multiple generations of training, playing off models against one another. """
    # TODO(cw/cd): add automatic storing of metadata with the models

    def __init__(self, hex_size=8, generations=10, bots_per_generation=5, start_bots=None, start_opponent_policies=None):
        self.hex_size = hex_size

        self.generations = generations
        self.bots_per_generation = bots_per_generation

        if start_bots is None:
            self.bots = [
                HexBotBrain(hex_size=self.hex_size)   # adjust here
                for _ in range(self.bots_per_generation)
            ]
        else:
            self.bots = start_bots

        self.opponent_policies = start_opponent_policies

    def play1v1(self, bot1: HexBotBrain, bot2: HexBotBrain, nr_games: int = 1000):
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

    def evolute(self, folder="test"):
        opponent_pool = []  # start playing against random policy
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")

            scores = [0. for _ in range(self.bots_per_generation)]
            for idx, bot in enumerate(self.bots):
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
                botdata_folder = f"{folder}/botdata"
                os.makedirs(botdata_folder, exist_ok=True)
                filename = f"{botdata_folder}/gen{gen+1}bot{idx+1}"

                scores[idx] = metadata["score"]

                trainer.save_trainee(botname, filename, metadata)

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
            opponent_pool = [(1., bot.play_policy) for bot in self.bots]

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
    bot_evo.evolute()


if __name__ == "__main__":
    main()
