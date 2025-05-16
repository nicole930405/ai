import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from alphazero import AlphaZeroAI
from alphazero import AlphaZeroNet
import os

BOARD_SIZE = 8
EPOCHS = 10
BATCH_SIZE = 64
NUM_SELFPLAY_GAMES = 20  # 每輪訓練時的自我對弈局數
LEARNING_RATE = 0.001  # 明確設定學習率
CLIP_VALUE = 1.0  # 設定梯度裁剪值

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

def on_board(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def random_move(legal_moves):
    return random.choice(legal_moves)

def get_legal_moves(board, player):
    opponent = 3 - player
    legal_moves = []

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] != 0:
                continue
            # 檢查八個方向是否有夾擊
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not on_board(nx, ny) or board[nx][ny] != opponent:
                    continue
                # 沿方向繼續找己方棋子
                while True:
                    nx += dx
                    ny += dy
                    if not on_board(nx, ny):
                        break
                    if board[nx][ny] == 0:
                        break
                    if board[nx][ny] == player:
                        legal_moves.append((x, y))
                        break
                else:
                    continue
                break

    return legal_moves

def apply_move(board, move, player):
    x, y = move
    board[x][y] = player
    opponent = 3 - player

    # 翻轉棋子
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        to_flip = []
        while on_board(nx, ny) and board[nx][ny] == opponent:
            to_flip.append((nx, ny))
            nx += dx
            ny += dy
        if on_board(nx, ny) and board[nx][ny] == player:
            for fx, fy in to_flip:
                board[fx][fy] = player

def apply_move_with_return(board, move, player):
    x, y = move
    board[x][y] = player
    opponent = 3 - player

    flipped_positions = []

    # 翻轉棋子，並記錄所有被翻轉的位置
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        to_flip = []
        while on_board(nx, ny) and board[nx][ny] == opponent:
            to_flip.append((nx, ny))
            nx += dx
            ny += dy
        if on_board(nx, ny) and board[nx][ny] == player:
            for fx, fy in to_flip:
                board[fx][fy] = player
            flipped_positions.extend(to_flip)

    return flipped_positions

def return_one_piece(board, pos, player):
    # 把 pos 位置返還成對方棋子
    board[pos[0]][pos[1]] = 3 - player

def generate_selfplay_data(model):
    data = []
    for game_num in range(NUM_SELFPLAY_GAMES):
        board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        # 初始設置 - Othello 標準開局
        board[3][3], board[4][4] = 2, 2  # 白子
        board[3][4], board[4][3] = 1, 1  # 黑子
        current_player = 1  # 黑子先行
        states, policies, players = [], [], []
        
        pass_count = 0  # 連續 pass 計數器
        
        while pass_count < 2:  # 兩方都無法落子時結束
            legal_moves = get_legal_moves(board, current_player)
            if not legal_moves:
                # 無法落子，換對方
                pass_count += 1
                current_player = 3 - current_player
                continue
            
            pass_count = 0  # 重置連續 pass 計數器
            
            tensor = model.board_to_tensor(board, current_player)
            with torch.no_grad():
                logits, _ = model.model(tensor)
                policy = logits.view(BOARD_SIZE, BOARD_SIZE).cpu().numpy()

            # 遮蔽非法落子並處理防止 NaN 問題
            masked = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for x, y in legal_moves:
                masked[x][y] = max(policy[x][y], 1e-8)  # 避免負值或極小值
            
            # 確保總和為 1，防止 NaN
            masked_sum = np.sum(masked)
            if masked_sum > 0:
                masked /= masked_sum
            else:
                # 如果總和為 0，使用均勻分布
                for x, y in legal_moves:
                    masked[x][y] = 1.0 / len(legal_moves)
            
            # 依據遮蔽後的策略選擇移動
            move_weights = [masked[x][y] for x, y in legal_moves]
            try:
                move = random.choices(legal_moves, weights=move_weights)[0]
            except ValueError:
                # 如果選擇失敗，使用均勻隨機選擇
                move = random.choice(legal_moves)
                
            states.append(model.board_to_tensor(board, current_player).squeeze(0).cpu().numpy())
            policies.append(masked)
            players.append(current_player)

            flipped = apply_move_with_return(board, move, current_player)

            # 還棋邏輯：吃超過兩顆需還一顆
            if len(flipped) > 2:
                return_one_piece(board, flipped[0], current_player)  # 返還其中一顆

            current_player = 3 - current_player

        # 計算勝負：1 = 黑勝, -1 = 白勝, 0 = 平手
        black_count = sum(row.count(1) for row in board)
        white_count = sum(row.count(2) for row in board)
        if black_count > white_count:
            winner = 1
        elif black_count < white_count:
            winner = 2
        else:
            winner = 0

        # 依據勝者分配獎勵值
        for i in range(len(states)):
            # 如果是勝者，value 為 1；如果是敗者，value 為 -1；平局為 0
            if winner == 0:
                value = 0
            else:
                value = 1 if players[i] == winner else -1
            data.append((states[i], policies[i], value))

    return data

def train_model(data, save_path="model.pt"):
    model = AlphaZeroNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 先檢查是否有現有模型，如果有則載入
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"已載入現有模型 {save_path}")
        except Exception as e:
            print(f"載入模型失敗: {e}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # 加入 weight decay 防止過擬合
    
    # 交叉熵用於策略，MSE 用於值函數
    # 使用交叉熵而非 KLDivLoss 防止數值問題
    loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_value = nn.MSELoss()

    # 添加學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    valid_batch_count = 0
    
    # 確保 data 不為空
    if not data:
        print("沒有生成任何訓練資料，請檢查 selfplay 過程")
        return
    
    print(f"開始訓練，共 {len(data)} 資料點")
    
    for epoch in range(EPOCHS):
        random.shuffle(data)
        total_loss = 0
        batch_count = 0

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            if not batch:
                continue
                
            try:
                # 建立 batch
                state_batch = torch.from_numpy(np.array([x[0] for x in batch])).float().to(device)
                
                # 將 policy 轉換為 index 形式，適合 CrossEntropyLoss
                policy_targets = []
                for x in batch:
                    policy = x[1]
                    # 找出最大概率的位置
                    flat_idx = np.argmax(policy.flatten())
                    policy_targets.append(flat_idx)
                
                policy_targets = torch.tensor(policy_targets, dtype=torch.long).to(device)
                value_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1).to(device)

                # 預測
                pred_policy, pred_value = model(state_batch)
                
                # 計算損失
                loss_policy = loss_fn_policy(pred_policy, policy_targets)
                loss_value = loss_fn_value(pred_value, value_batch)
                loss = loss_policy + loss_value

                # 檢查是否有 NaN 值
                if torch.isnan(loss).any():
                    print(f"警告：批次 {i//BATCH_SIZE} 中檢測到 NaN 損失，跳過此批次")
                    continue
                
                total_loss += loss.item()
                batch_count += 1

                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
                
                optimizer.step()
                valid_batch_count += 1
                
            except Exception as e:
                print(f"處理批次時出錯: {e}")
                continue

        # 計算平均損失
        avg_loss = total_loss / max(1, batch_count)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        # 更新學習率
        scheduler.step(avg_loss)
        
        # 每個 epoch 結束後儲存模型
        torch.save(model.state_dict(), f"{save_path}.epoch{epoch+1}")
    
    # 儲存最終模型
    torch.save(model.state_dict(), save_path)
    print(f"模型已儲存至 {save_path}")
    print(f"成功處理的批次總數: {valid_batch_count}")


if __name__ == "__main__":
    # 確保 AlphaZeroAI 可以正常初始化
    try:
        # 嘗試建立新的 AI 物件
        if os.path.exists("model.pt"):
            alphazero = AlphaZeroAI(model_path="model.pt")
        else:
            # 如果模型不存在，從頭開始
            alphazero = AlphaZeroAI()
        
        print("生成自我對弈資料中...")
        data = generate_selfplay_data(alphazero)
        
        if data:
            print(f"生成了 {len(data)} 筆訓練資料")
            train_model(data)
        else:
            print("未生成任何訓練資料，請檢查自我對弈邏輯")
            
    except Exception as e:
        print(f"初始化 AlphaZeroAI 時發生錯誤: {e}")