import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import time

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        self.model_loaded = False
        
        # Input shape: (batch_size, 3, 8, 8)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64)  # output = 8x8 board policy
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

     # 添加board_to_tensor方法，處理8x8棋盤轉換為輸入張量
    def board_to_tensor(self, board, current_player):
        # 創建三個特徵平面：當前玩家、對手、空位
        features = np.zeros((3, 8, 8), dtype=np.float32)
        
        opponent = 3 - current_player
        
        for i in range(8):
            for j in range(8):
                if board[i][j] == current_player:
                    features[0, i, j] = 1  # 當前玩家
                elif board[i][j] == opponent:
                    features[1, i, j] = 1  # 對手
                else:
                    features[2, i, j] = 1  # 空位
        
        # 轉換為PyTorch張量並添加批次維度
        tensor = torch.from_numpy(features).float().unsqueeze(0)
        
        # 如果有device定義，移至對應設備
        if hasattr(self, 'device'):
            tensor = tensor.to(self.device)
        
        return tensor
    
    def predict(self, board, current_player):
        """
        board: numpy array with shape (8,8)
        return: policy (8x8), value (scalar)
        """
        tensor = self.board_to_tensor(board, current_player)
        with torch.no_grad():
            policy, value = self.forward(tensor)
        return policy.squeeze(0).numpy(), value.item()

    #def _board_to_tensor(self, board):
        """
        Converts the 8x8 board to a tensor of shape (1, 2, 8, 8)
        where [0] is player's stones and [1] is opponent's stones
        """
        p1 = (board == 1).astype(np.float32)
        p2 = (board == 2).astype(np.float32)
        tensor = np.stack([p1, p2], axis=0)  # Shape: (2, 8, 8)
        return torch.tensor(tensor).unsqueeze(0)  # Shape: (1, 2, 8, 8)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.eval()
        self.model_loaded = True

class MCTS:
    """Monte Carlo Tree Search for AlphaZero."""
    def __init__(self, model, simulations=100):
        self.model = model
        self.simulations = simulations
    
    def search(self, state, player):
        """Perform MCTS search and return the best move."""
        # For simplicity, just choose a smart move
        return self._choose_smart_move(state, player)
    
    def _choose_smart_move(self, board, player):
        """Choose a smart move based on board positions without full MCTS."""
        # Priority corners, then edges, avoid cells adjacent to corners
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        bad_cells = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7), 
                     (6, 0), (6, 1), (7, 1), (6, 6), (6, 7), (7, 6)]
        
        # Get valid moves
        valid_moves = [(r, c) for r in range(8) for c in range(8) 
                      if board[r][c] == 0 and self._is_valid_move(board, r, c, player)]
        
        if not valid_moves:
            return None
        
        # First try corners
        for move in corners:
            if move in valid_moves:
                return move
        
        # Then try edges but not bad cells
        edges = [(i, j) for i in range(8) for j in range(8) 
                if (i == 0 or i == 7 or j == 0 or j == 7) and (i, j) not in bad_cells]
        edge_moves = [move for move in valid_moves if move in edges]
        if edge_moves:
            return random.choice(edge_moves)
        
        # Avoid bad cells if possible
        good_moves = [move for move in valid_moves if move not in bad_cells]
        if good_moves:
            return random.choice(good_moves)
        
        # If all else fails, choose a random move
        return random.choice(valid_moves)
    
    def _is_valid_move(self, board, row, col, player):
        """Check if move is valid."""
        if board[row][col] != 0:
            return False
            
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
                      
        opponent = 2 if player == 1 else 1
        
        for dx, dy in directions:
            x, y = row + dx, col + dy
            count = 0
            while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
                x += dx
                y += dy
                count += 1
            if count > 0 and 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
                return True
        return False

class AlphaZeroAI:
    """A simple AI using AlphaZero principles for Reversi."""
    def __init__(self, model_path=None):
        self.net = AlphaZeroNet()
        if model_path and os.path.exists(model_path):
            self.net.load_model(model_path)
            self.trained = True
        else:
            self.trained = False

        self.mcts = MCTS(model=self.net, simulations=100)
        
        # If we don't have a trained model, use a simple strategy
        self.trained = self.net.model_loaded
    
    def choose_move(self, board, player, legal_moves):
        """Choose a move using either MCTS or a simple strategy."""
        if not legal_moves:
            return None
        
        # Add a small delay to simulate thinking
        time.sleep(0.5)
        
        # Use MCTS if we have a trained model
        if self.trained:
            move = self.mcts.search(board, player)
            if move in legal_moves:
                return move
        
        # Fall back to a simpler strategy
        # Prioritize corners
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for move in corners:
            if move in legal_moves:
                return move
        
        # Then edges
        edges = [(i, 0) for i in range(1, 7)] + [(i, 7) for i in range(1, 7)] + \
                [(0, i) for i in range(1, 7)] + [(7, i) for i in range(1, 7)]
        edge_moves = [move for move in legal_moves if move in edges]
        if edge_moves:
            return random.choice(edge_moves)
        
        # Then maximize flips
        best_move = None
        max_flips = -1
        
        for move in legal_moves:
            flips = self._count_flips(board, move[0], move[1], player)
            if flips > max_flips:
                max_flips = flips
                best_move = move
        
        return best_move
    
    def _count_flips(self, board, row, col, player):
        """Count how many pieces would be flipped with this move."""
        if board[row][col] != 0:
            return 0
            
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
                      
        opponent = 2 if player == 1 else 1
        total_flips = 0
        
        for dx, dy in directions:
            x, y = row + dx, col + dy
            flips = 0
            while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
                flips += 1
                x += dx
                y += dy
            if flips > 0 and 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
                total_flips += flips
        
        return total_flips