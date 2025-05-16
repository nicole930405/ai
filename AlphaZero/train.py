import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import AlphaZeroNet  # 你已有的模型結構
from game import ReversiGame    # 你要寫的遊戲邏輯，用來模擬對局
from mcts import MCTS           # 你已有的 MCTS

# 訓練超參數
EPOCHS = 10
BATCH_SIZE = 64
SIMULATIONS = 50
BUFFER_SIZE = 10000
TRAIN_AFTER = 1000  # 幾筆資料後才開始訓練
LEARNING_RATE = 0.001

# 訓練儲存區
replay_buffer = deque(maxlen=BUFFER_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

def self_play_game():
    game = ReversiGame()
    states, policies, players = [], [], []

    while not game.is_game_over():
        board = game.get_board()
        current_player = game.get_current_player()
        tensor_input = game.board_to_tensor(board, current_player)

        mcts = MCTS(model, simulations=SIMULATIONS)
        move_probs = mcts.run_return_probs(board, current_player, game.get_valid_moves)

        move = np.unravel_index(np.argmax(move_probs), (8, 8))
        states.append(tensor_input)
        policies.append(move_probs)
        players.append(current_player)

        game.apply_move(move, current_player)

    winner = game.get_winner()
    data = []
    for state, policy, player in zip(states, policies, players):
        value = 1 if player == winner else -1 if winner != 0 else 0
        data.append((state, policy, value))
    return data

def train():
    if len(replay_buffer) < TRAIN_AFTER:
        return

    batch = random.sample(replay_buffer, BATCH_SIZE)
    state_batch = torch.stack([s for (s, _, _) in batch]).to(device)
    policy_batch = torch.tensor([p for (_, p, _) in batch], dtype=torch.float32).to(device)
    value_batch = torch.tensor([v for (_, _, v) in batch], dtype=torch.float32).unsqueeze(1).to(device)

    model.train()
    pred_policy, pred_value = model(state_batch)
    loss_policy = nn.CrossEntropyLoss()(pred_policy, policy_batch.argmax(dim=1))
    loss_value = loss_fn(pred_value, value_batch)
    loss = loss_policy + loss_value

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        game_data = self_play_game()
        replay_buffer.extend(game_data)
        loss = train()
        if loss:
            print(f"Training loss: {loss:.4f}")
        torch.save(model.state_dict(), "model.pt")
