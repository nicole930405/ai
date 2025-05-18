# reversi_trainer.py
# 用於產生訓練資料（模仿學習）與訓練神經網路模型

import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import ai 
import time

DATA_FILE = "reversi_data.npy"
MODEL_FILE = "reversi_policy_model.h5"

# 產生棋盤資料（AI 對弈資料集）
def generate_training_data(num_games=1, save_every=1):
    data = []
    for i in range(num_games):
        print(f"開始第 {i+1} 局")
        board = [[0 for _ in range(8)] for _ in range(8)]
        board[3][3], board[3][4] = 1, 2
        board[4][3], board[4][4] = 2, 1
        player = 1

        move_count = 0
        while True:
            moves = ai.get_valid_moves(board, player)
            if not moves:
                player = 3 - player
                if not ai.get_valid_moves(board, player):
                    break
                continue

            start = time.time()
            move = ai.get_best_move(board, player, max_depth=4, time_limit=0.5)
            if move is None:
                print("⚠️ 無法找到合法落子，跳過這局")
                break

            print(f"  ➤ 第 {move_count+1} 步, 玩家 {player}, 用時 {time.time() - start:.3f} 秒")

            board_array = np.array(board).astype(np.int8)
            move_array = np.zeros((8, 8), dtype=np.float32)
            move_array[move[0]][move[1]] = 1.0
            data.append((board_array, move_array))
            board = ai.apply_move(board, move, player)
            player = 3 - player
            move_count += 1

        print(f"✅ 第 {i+1} 局完成，共 {move_count} 步")
        if (i + 1) % save_every == 0:
            np.save(DATA_FILE, data)
            print(f"📁 資料已儲存：共 {len(data)} 筆")

    return data

# 建立模型（簡單 CNN）
def build_model():
    model = models.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='softmax'),  # flattened (8x8)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 訓練主程序
def train_model():
    if os.path.exists(DATA_FILE):
        print("載入已存在的訓練資料...")
        data = np.load(DATA_FILE, allow_pickle=True).tolist()
    else:
        print("產生訓練資料中...")
        data = generate_training_data(num_games=50)

    # ✅ 額外再產生一些資料（可選）
    data += generate_training_data(num_games=50)

    # 拆開資料
    boards, moves = zip(*data)
    boards = np.array(boards).reshape(-1, 8, 8, 1) / 2.0
    moves = np.array(moves).reshape(-1, 64)

    # ✅ 若模型存在就載入，並重新 compile
    if os.path.exists(MODEL_FILE):
        print("載入舊模型繼續訓練...")
        model = tf.keras.models.load_model(MODEL_FILE)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 必加這行
    else:
        print("建立新模型...")
        model = build_model()

    # 訓練
    print(f"開始訓練，共 {len(boards)} 筆資料")
    model.fit(boards, moves, epochs=10, batch_size=32, verbose=1)
    model.save(MODEL_FILE)
    print(f"✅ 模型已儲存為 {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
