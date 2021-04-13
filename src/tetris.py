import numpy as np
import matplotlib.pyplot as plt


class Tetris:
    colors: np.ndarray = np.array([
        [0, 0, 0],
        [255, 255, 0],
        [147, 88, 254],
        [54, 175, 144],
        [255, 0, 0],
        [102, 217, 238],
        [254, 151, 32],
        [0, 0, 255]
    ], dtype=np.uint8)

    pieces: np.ndarray = np.array([
        [[0, 0],   [0, 0], [0, 0],  [0, 0]],  # None
        [[0, -1],  [0, 0], [-1, 0], [-1, 1]],  # S
        [[0, -1],  [0, 0], [1, 0],  [1, 1]],  # Z
        [[0, -1],  [0, 0], [0, 1],  [0, 2]],  # I
        [[-1, 0],  [0, 0], [1, 0],  [0, 1]],  # T
        [[1, 0],   [0, 0], [0, 1],  [1, 1]],  # O
        [[-1, -1], [0, 0], [0, -1], [0, 1]],  # J
        [[1, -1],  [0, 0], [0, -1], [0, 1]]],  # L
    )

    def __init__(self, num_rows: int = 20, num_cols: int = 10, block_size: int = 20) -> None:
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self.block_size: int = block_size

        self.board: np.ndarray = np.empty((self.num_rows, self.num_cols), dtype=np.uint8)
        self.score: float = 0.0
        self.cleared_lines: int = 0
        self.shape: int = 0
        self.piece: np.ndarray = np.empty_like(self.pieces[0])
        self.position: dict = {"x": 0, "y": 0}
        self.gameover: bool = False

        self.last_value: float = 0.0

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.score = 0.0
        self.cleared_lines = 0
        self.new_piece()
        self.gameover = False
        self.last_value = 0.0
        return (self.board > 0).astype(float)

    def rotate(self, piece: np.ndarray) -> None:
        rotation_270 = np.array([[0, -1], [1, 0]])
        np.dot(piece, rotation_270, out=piece)

    # def get_state_properties(self, board):
    #     lines_cleared, board = self.clear_lines(board)
    #     holes = self.get_holes(board)
    #     bumpiness, height = self.get_bumpiness_and_height(board)

    #     return [lines_cleared, holes, bumpiness, height]

    # def get_holes(self, board):
    #     num_holes = 0
    #     for col in zip(*board):
    #         row = 0
    #         while row < self.height and col[row] == 0:
    #             row += 1
    #         num_holes += len([x for x in col[row + 1:] if x == 0])
    #     return num_holes

    # def get_bumpiness_and_height(self, board):
    #     board = np.array(board)
    #     mask = board != 0
    #     invert_heights = np.where(
    #         mask.any(axis=0), np.argmax(mask, axis=0), self.height)
    #     heights = self.height - invert_heights
    #     total_height = np.sum(heights)
    #     currs = heights[:-1]
    #     nexts = heights[1:]
    #     diffs = np.abs(currs - nexts)
    #     total_bumpiness = np.sum(diffs)
    #     return total_bumpiness, total_height

    def get_num_holes(self) -> int:
        exists_piece = self.board != 0
        top_indicies = np.where(
            np.any(exists_piece, axis=0), np.argmax(exists_piece, axis=0), self.num_rows)
        row_indicies = np.repeat(
            np.arange(self.board.shape[0])[:, np.newaxis],
            self.board.shape[1],
            axis=1)
        below_top = row_indicies > top_indicies

        num_holes = np.sum(np.logical_and(np.logical_not(exists_piece), below_top))
        return num_holes

    def get_max_height(self) -> int:
        exists_piece = self.board != 0
        invert_heights = np.where(
            np.any(exists_piece, axis=0), np.argmax(exists_piece, axis=0), self.num_rows)
        heights = self.num_rows - invert_heights
        max_height = np.max(heights)
        return max_height

    def get_total_bumpiness(self) -> int:
        exists_piece = self.board != 0
        invert_heights = np.where(
            np.any(exists_piece, axis=0), np.argmax(exists_piece, axis=0), self.num_rows)
        heights = self.num_rows - invert_heights
        diffs = np.abs(heights[:-1] - heights[1:])
        total_bumpiness = np.sum(diffs)
        return total_bumpiness

    def get_next_states(self) -> dict:
        states = {}
        piece = self.piece.copy()
        position = {"x": 0, "y": 0}
        assert self.shape != 0
        if self.shape == 5:
            num_rotations = 1
        elif self.shape in [1, 2, 3]:
            num_rotations = 2
        else:
            num_rotations = 4

        for num_rotation in range(num_rotations):
            for x in range(-piece[:, 0].min(), self.num_cols - piece[:, 0].max()):
                position["x"] = x
                position["y"] = -piece[:, 1].min()
                if self.check_collision(piece, position):
                    continue

                while not self.check_collision(piece, position):
                    position["y"] += 1
                position["y"] -= 1

                board = self.board.copy()
                self.store(piece, self.shape, position, board)
                states[(x, num_rotation)] = (board > 0).astype(float)
            self.rotate(piece)
        return states

    # def get_current_board_state(self):
    #     board = [x[:] for x in self.board]
    #     for y in range(self.piece.shape[0]):
    #         for x in range(len(self.piece[y])):
    #             board[y + self.position["y"]][x +
    #                                           self.position["x"]] = self.piece[y][x]
    #     return board

    def new_piece(self) -> bool:
        self.shape = np.random.randint(0, 7) + 1
        np.copyto(self.piece, self.pieces[self.shape])
        self.position["x"] = self.num_cols // 2 - 1
        self.position["y"] = -self.piece[:, 1].min()

        self.gameover = self.check_collision(self.piece, self.position)

        return self.gameover

    def check_collision(self, piece: np.ndarray, position: dict) -> bool:
        next_xs = position['x'] + piece[:, 0]
        next_ys = position['y'] + piece[:, 1]
        return np.any(next_xs < 0) or np.any(self.num_cols <= next_xs) or \
            np.any(next_ys < 0) or np.any(self.num_rows <= next_ys) or \
            np.any(self.board[next_ys, next_xs]) != 0

    def store(self, piece: np.ndarray, shape: int, position: dict, board: np.ndarray) -> None:
        xs = position["x"] + piece[:, 0]
        ys = position["y"] + piece[:, 1]
        board[ys, xs] = shape

    def clear_lines(self, board: np.ndarray) -> int:
        num_cleared_lines = 0
        for y in range(self.num_rows):
            if all(board[y] != 0):
                for i in range(y, 0, -1):
                    np.copyto(board[i], board[i - 1])
                num_cleared_lines = num_cleared_lines + 1
        return num_cleared_lines

    def step(self, action: tuple) -> tuple:
        x, num_rotations = action
        for _ in range(num_rotations):
            self.rotate(self.piece)
        self.position["x"] = x
        self.position["y"] = -self.piece[:, 1].min()

        while not self.check_collision(self.piece, self.position):
            self.position["y"] += 1
        self.position["y"] -= 1

        self.store(self.piece, self.shape, self.position, self.board)

        num_cleared_lines = self.clear_lines(self.board)

        # current_value = - 0.5 * self.get_max_height() \
            # - 0.1 * self.get_num_holes() \
            # - 0.25 * self.get_total_bumpiness()
        # reward = current_value - self.last_value + num_cleared_lines ** 2 + 1
        reward = num_cleared_lines + 1
        # self.last_value = current_value

        self.score += reward
        self.cleared_lines += num_cleared_lines
        gameover = self.new_piece()
        if gameover:
            reward -= 2
        return reward, gameover

    def render(self) -> np.ndarray:
        board = self.board.copy()
        if not self.gameover:
            self.store(self.piece, self.shape, self.position, board)
        return self.colors[board]


if __name__ == "__main__":
    env = Tetris()
    env.reset()

    fig = plt.figure()
    img = plt.imshow(env.render())
    fig.suptitle('step: 0')
    step = 0
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        for key in next_steps.keys():
            print(key)
            print(next_steps[key])
        # print(next_steps)
        action = next_actions[np.random.randint(len(next_actions))]
        reward, done = env.step(action)
        print(f"reward {reward}")

        img.set_data(env.render())
        fig.suptitle('step: {}'.format(step + 1))
        step = step + 1
        plt.axis('off')
        plt.pause(0.5)

        if done:
            break
    plt.show()
