import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import copy
from collections import defaultdict

BOARD_SIZE = 8  # 黑白棋為 8x8

# 簡化版神經網路
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.fc_value = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        policy = self.fc_policy(x)  # 對每一格評分
        value = torch.tanh(self.fc_value(x))  # 預測勝率（-1 ~ 1）
        return policy, value

# AlphaZero AI 包裝類別
class AlphaZeroAI:
    def __init__(self, model_path="model.pt"):
        self.model = AlphaZeroNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 預設為推論模式
        self.model_path = model_path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(" AlphaZero 模型已載入")
        else:
            print(" 尚未訓練模型，使用未訓練狀態")

    def board_to_tensor(self, board, color):
        """
        將棋盤轉為神經網路輸入格式（2 x 8 x 8 tensor）：
        - 第 0 層是我方棋子
        - 第 1 層是敵方棋子
        """
        me = (np.array(board) == color).astype(np.float32)
        opp = (np.array(board) == (3 - color)).astype(np.float32)
        tensor = np.stack([me, opp], axis=0)
        return torch.tensor(tensor).unsqueeze(0).to(self.device)

    def choose_move(self, board, color, legal_moves):
        """
        使用 AlphaZero 模型推論落子位置
        """
        if not legal_moves:
            return None

        input_tensor = self.board_to_tensor(board, color)
        with torch.no_grad():
            policy_logits, _ = self.model(input_tensor)
            policy = policy_logits.view(BOARD_SIZE, BOARD_SIZE).cpu().numpy()

        # 在合法落子中選取分數最高的
        best_move = None
        best_score = -float("inf")
        for move in legal_moves:
            x, y = move
            if policy[x][y] > best_score:
                best_score = policy[x][y]
                best_move = move

        return best_move

class MCTS:
    def __init__(self, model, simulations=100, c_puct=1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.Q = {}  # Q 值
        self.N = defaultdict(int)  # 訪問次數
        self.P = {}  # policy priors

    def run(self, board, player, get_valid_moves):
        for _ in range(self.simulations):
            self.search(board, player, get_valid_moves)

    def search(self, board, player, get_valid_moves):
        key = self.serialize_board(board, player)

        if key not in self.P:
            tensor = self.model.board_to_tensor(board, player)
            with torch.no_grad():
                policy_logits, value = self.model.model(tensor)
            policy = policy_logits.view(BOARD_SIZE, BOARD_SIZE).cpu().numpy()
            legal_moves = get_valid_moves(board, player)
            policy = self.mask_illegal(policy, legal_moves)

            self.P[key] = policy
            self.N[key] = 0
            return value.item()

        best_ucb, best_move = -float("inf"), None
        total_n = sum(self.N[(key, move)] for move in self.get_legal_moves(board, player, get_valid_moves))
        for move in self.get_legal_moves(board, player, get_valid_moves):
            move_key = (key, move)
            q = self.Q.get(move_key, 0)
            n = self.N.get(move_key, 0)
            u = self.c_puct * self.P[key][move[0]][move[1]] * math.sqrt(total_n + 1) / (1 + n)
            ucb = q + u
            if ucb > best_ucb:
                best_ucb = ucb
                best_move = move

        next_board = copy.deepcopy(board)
        self.apply_move(next_board, best_move, player)
        next_player = 3 - player
        v = -self.search(next_board, next_player, get_valid_moves)

        move_key = (key, best_move)
        self.N[move_key] += 1
        self.Q[move_key] = (self.Q.get(move_key, 0) * (self.N[move_key] - 1) + v) / self.N[move_key]
        self.N[key] += 1
        return v

    def choose_move(self, board, player, get_valid_moves):
        self.run(board, player, get_valid_moves)
        key = self.serialize_board(board, player)
        legal_moves = self.get_legal_moves(board, player, get_valid_moves)
        max_visits = -1
        best_move = None
        for move in legal_moves:
            visits = self.N.get((key, move), 0)
            if visits > max_visits:
                max_visits = visits
                best_move = move
        return best_move

    def mask_illegal(self, policy, legal_moves):
        mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for x, y in legal_moves:
            mask[x][y] = 1
        policy *= mask
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        return policy

    def get_legal_moves(self, board, player, get_valid_moves):
        return get_valid_moves(board, player)

    def apply_move(board, move, player):
        x, y = move
        board[x][y] = player
        # You should apply flipping here as well

    def serialize_board(self, board, player):
        return tuple(map(tuple, board)), player