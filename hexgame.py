import random
from typing import Callable, List, Optional, Tuple, Set
import pygame
import numpy as np


class HexGame:
    """ A barebones implementation of the game Hex with minimalist rendering. """
    EMPTY, RED, BLUE = [0, 1, 2]
    PLAYER_COLOR = RED
    OPPONENT_COLOR = BLUE

    def __init__(
            self,
            size: int = 5,
            start_color=RED,
            opponent_policy: Optional[Callable] = None,
            opponent_pool: List[Tuple[float, Callable]] = None,
            auto_reset: bool = False,
            render_enabled: bool = False,
            render_fps: int = 2):
        """
        Initialises Hex Game as a gymnasium environment.

        Keywords:
        size: integer setting the sidelength of the hex arena
        start_color: indicates which color starts, should be one of \
            Hex_Game.EMPTY, Hex_Game.RED, Hex_Game.BLUE
        opponent_policy: an Optional function executing taking the game \
            state and returning the opponent action \
            None: sample opponents from opponent_pool, if specified \
            if opponent_pool is None, use the random policy
        opponent_pool: a list of tuples (weight, policy) of weighted \
            opponent policies if opponent_policy is set to "pool"
        auto_reset: if True, automatically restarts a new game after ending
        render_enabled: if True, render every frame, which slows play
        render_fps: fps of displayed games
        """
        # Game statics initilization
        self.size = size
        self.state = np.array([[HexGame.EMPTY for _ in range(size)]
                               for _ in range(size)])
        self.free_tiles = list(range(self.size * self.size))
        self.start_color = start_color
        self.player_color = HexGame.PLAYER_COLOR
        self.opponent_color = HexGame.OPPONENT_COLOR
        self.auto_reset = auto_reset

        # Game dynamics (policy) initialization
        if (opponent_pool) and (opponent_policy is None):
            self.opponent_pool = opponent_pool
            self.opponent_policy = self.pick_pool_policy()
        elif (opponent_policy):
            self.opponent_pool = None
            self.opponent_policy = opponent_policy
            if (opponent_pool):
                print(
                    "Both opponent_policy and opponent_pool specified. Ignoring opponent_pool.")
        else:
            self.opponent_pool = None
            self.opponent_policy = self.rand_policy

        # Board index lookup lists
        self._action_to_hexagon = [
            divmod(idx, self.size) for idx in range(self.size * self.size)
        ]
        self._transpose_action = [
            (idx % self.size)*self.size + (idx // self.size) for idx in range(self.size * self.size)
        ]

        # Render stuff
        self.window = None
        self.clock = None
        self.window_size = 200
        self.render_enabled = render_enabled
        self.render_fps = render_fps

        # If opponent plays first, make that move
        if self.start_color == self.opponent_color:
            self.opponent_play()

    """
    +----------------+
    | POLICY METHODS |
    +----------------+
    """

    def pick_pool_policy(self):
        """
        Picks a policy from self.opponent_pool according to the given weights. 
        """
        total_weight = sum(weight for (weight, _) in self.opponent_pool)
        opponent_decider = random.random() * total_weight
        cumulative_weight = 0
        for weight, policy in self.opponent_pool:
            cumulative_weight += weight
            if cumulative_weight > opponent_decider:
                return policy
        print("Error: no policy selected.")
        return None

    def rand_policy(self, state, free_tiles):
        """ 
        The random policy which simply selects a random free tile.
        """
        rand_free_tile_index = random.randint(0, len(free_tiles)-1)
        return free_tiles[rand_free_tile_index]

    """
    +----------------------+
    | GAME STATICS METHODS |
    +----------------------+
    """

    def flat_state(self, transpose_roles=False):
        """
        Getter function returning the flattened state of the game.
        If transpose_roles=True, then transposed state (switched colors, 
        90-deg rotated board) is returned.
        """
        state = np.zeros((self.size, self.size, 2), dtype=np.float32)

        for x in range(self.size):
            for y in range(self.size):
                if self.state[x, y] == 0:
                    continue
                else:
                    # Recall RED = 1 and BLUE = 2
                    color = self.state[y, x]-1
                    # Switch color if transpose_roles
                    if transpose_roles:
                        color = (color + 1) % 2
                    state[x, y, color] = 1
                # For 3-Hot encoding use this instead:
                # state[x,y,self.state[y,x]] = 1
        # Rotate board if transpose_roles
        if transpose_roles:
            state = np.swapaxes(state, 0, 1)
        return state.reshape(self.size * self.size * 2)

    def neighbours(self, row: int, col: int) -> List[Tuple[int]]:
        """
        Return the neighbours of (row,col) in a list
        """
        return [
            (r, c) for (r, c) in
            [(row-1, col), (row-1, col+1), (row, col-1),
                (row, col+1), (row+1, col-1), (row+1, col)]
            if (r >= 0 and r < self.size and c >= 0 and c < self.size)
        ]

    def borders_reached_from_tile(
            self, row: int,
            column: int,
            color: int,
            visited: Optional[Set] = None
    ) -> Tuple[bool]:
        """
        Return a pair of bools indicating whether the two
        borders corresponding to 'color' have been reached.

        Keywords:
        row, column: specify the hex tile
        color: color of the tile placed at this hex tile
        visited: auxiliary argument for DFS

        Returns a pair of bools:
        border1: bool indicating if row/col 0 is reachable for
            color RED/BLUE
        border2: bool indicating if row/col `size` is reachable for
            color RED/BLUE
        """
        if visited is None:
            visited = set()
        visited.add((row, column))
        neibs = [neib for neib in self.neighbours(
            row, column) if neib not in visited]

        border1 = (row == 0 and color == HexGame.RED) or (
            column == 0 and color == HexGame.BLUE)
        border2 = (row == self.size-1 and color == HexGame.RED) or (
            column == self.size-1 and color == HexGame.BLUE)
        for r, c in neibs:
            if self.state[c, r] == color:
                b1, b2 = self.borders_reached_from_tile(r, c, color, visited)
                border1 = border1 or b1
                border2 = border2 or b2
        return border1, border2

    """
    +-----------------------+
    | GAME DYNAMICS METHODS |
    +-----------------------+
    """

    def play_tile(self, action: int, color: int) -> bool:
        """
        Play a tile of the color at the specified location.

        Keywords:
        row: the row (between 0 and self.size)
        column: the column (between 0 and self.size)
        color: the color (RED or BLUE)

        Returns a bool indicating whether this play won the game
        for the given color.
        """
        row, column = self._action_to_hexagon[action]

        self.free_tiles.remove(action)

        self.state[column, row] = color

        border1, border2 = self.borders_reached_from_tile(
            row, column, color)

        return border1 and border2

    def play_out(self, player_policy: Callable) -> bool:
        """ 
        Play out the rest of the game by repeatedly calling
        player_policy. Starts with a player move.

        Keywords:
        player_policy: a callable function taking state and free_tiles as input
            and returning a move

        Return True if the player wins, and False otherwise.
        """
        terminated = False
        while not terminated:
            action = player_policy(self.flat_state(), self.free_tiles)
            _, reward, terminated = self.step(action)

        if reward > 0:
            return True

        return False

    def opponent_play(self):
        """
        Execute opponent policy. Return whether opponent wins.
        ASSUMPTION: opponent looks at tranposed, color-inverted board
        """
        transposed_state = self.flat_state(transpose_roles=True)
        transposed_free_tile = [self._transpose_action[idx]
                                for idx in self.free_tiles]
        transposed_opponent_action = self.opponent_policy(
            transposed_state, transposed_free_tile)
        opponent_action = self._transpose_action[transposed_opponent_action]
        return self.play_tile(opponent_action, self.opponent_color)

    def reset(self):
        """Reset the game state."""
        self.state = np.array([[HexGame.EMPTY for _ in range(self.size)]
                               for _ in range(self.size)])
        self.free_tiles = list(range(self.size * self.size))

        if self.opponent_pool:
            self.opponent_policy = self.pick_pool_policy()

        if self.start_color == HexGame.BLUE:
            self.opponent_play()

    def step(self, action):
        """
        Step the environment given the current action.
        Both the action and an opponent action (if possible)
        are executed.

        Keywords:
        action: a number between 0 and self.size ** 2 -1

        Returns a tuple consisting of:
        new_state: state after the action of player and opponent. \
            if auto_reset is enabled and the game ended, this is the 
            state in the new game
        reward: 1 if the game was won, -1 if it was lost, else 0
        terminated: True if the game ended, False otherwise
        """

        # Overwrite action if optimal 2x2 policy activated
        if self.size == 2 and self.optimal_2x2_play == True:
            action = self.optimal_2x2_policy()

        terminated = self.play_tile(action, self.player_color)

        new_state = self.state

        reward = 0
        if terminated:
            reward = 1
        else:
            terminated = self.opponent_play()
            if terminated:
                reward = -1

        if self.auto_reset and terminated:
            self.reset()

        if self.render_enabled:
            self._render_frame()

        return new_state, reward, terminated

    def _render_frame(self):
        if self.window is None and self.render_enabled:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_enabled:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Add gridlines indicating allowed connections.
        for x in range(self.size):
            pygame.draw.line(  # horizontal lines
                canvas,
                0,
                (pix_size * 0.5, pix_size * (x + 0.5)),
                (self.window_size-pix_size*0.5, pix_size * (x + 0.5)),
                width=3,
            )
            pygame.draw.line(  # vertical lines
                canvas,
                0,
                (pix_size * (x + 0.5), pix_size * 0.5),
                (pix_size * (x + 0.5), self.window_size-pix_size*0.5),
                width=3,
            )
            pygame.draw.line(  # diagonals part 1
                canvas,
                0,
                (pix_size * (x + 0.5), pix_size * 0.5),
                (pix_size * 0.5, pix_size * (x + 0.5)),
                width=3,
            )
            pygame.draw.line(  # diagonals part 2
                canvas,
                0,
                (self.window_size-pix_size * (x + 0.5),
                    self.window_size-pix_size * 0.5),
                (self.window_size-pix_size * 0.5,
                    self.window_size-pix_size * (x + 0.5)),
                width=3,
            )
        # Draw lines on either side indicating the goals for each color
            pygame.draw.polygon(  # RED
                canvas,
                (255, 0, 0),
                [(0, 0), (pix_size/5, pix_size/5),
                    (pix_size/5, self.window_size-pix_size/5),
                    (0, self.window_size)]
            )
            pygame.draw.polygon(  # RED
                canvas,
                (255, 0, 0),
                [(self.window_size, 0), (self.window_size, self.window_size),
                    (self.window_size-pix_size/5, self.window_size-pix_size/5),
                    (self.window_size-pix_size/5, pix_size/5)]
            )
            pygame.draw.polygon(  # BLUE
                canvas,
                (0, 0, 255),
                [(0, 0), (self.window_size, 0),
                    (self.window_size-pix_size/5, pix_size/5),
                    (pix_size/5, pix_size/5)]
            )
            pygame.draw.polygon(  # BLUE
                canvas,
                (0, 0, 255),
                [(0, self.window_size),
                    (pix_size/5, self.window_size-pix_size/5),
                    (self.window_size-pix_size/5, self.window_size-pix_size/5),
                    (self.window_size, self.window_size)]
            )
        # We draw all placed tiles (draw them circular?)
        for column in range(self.size):
            for row in range(self.size):
                if self.state[column, row] == HexGame.RED:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        ((row+0.5)*pix_size, (column+0.5)*pix_size),
                        pix_size/4
                    )
                elif self.state[column, row] == HexGame.BLUE:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 255),
                        ((row+0.5)*pix_size, (column+0.5)*pix_size),
                        pix_size/4
                    )

        if self.render_enabled:
            # Copy our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Make sure human-rendering occurs at predefined framerate.
            # Automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


def test():
    """Runs HexGame, playing random vs. random."""
    size = 5
    start_color = HexGame.RED  # AI goes first
    hg = HexGame(size, start_color, auto_reset=False, optimal_2x2_play=True)
    terminated = False
    while not terminated:
        random_action = hg.free_tiles[random.randint(0, len(hg.free_tiles)-1)]
        _, _, terminated = hg.step(random_action)
    input("Press Enter to quit.")


if __name__ == "__main__":
    test()
