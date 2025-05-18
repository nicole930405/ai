import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from alphazero import AlphaZeroAI
from alphazero import AlphaZeroNet
import os
import sys
import io
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

BOARD_SIZE = 8
EPOCHS = 30
BATCH_SIZE = 64
NUM_SELFPLAY_GAMES = 50  # 每輪訓練時的自我對弈局數
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

def board_to_tensor(board, current_player):
    # 創建兩個特徵平面，一個用於當前玩家的棋子，一個用於對手的棋子
    features = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    opponent = 3 - current_player
    
    # 設置棋盤狀態
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == current_player:
                features[0, i, j] = 1  # 當前玩家的棋子
            elif board[i][j] == opponent:
                features[1, i, j] = 1  # 對手的棋子
            else:
                features[2, i, j] = 1  # 空位
    
    # 轉換為 PyTorch 張量並添加批次維度
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    return tensor

def generate_selfplay_data(model):
    try:
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
                
                # 檢查模型有哪些方法可用
                if hasattr(model, 'board_to_tensor'):
                    # 如果模型有此方法，直接呼叫
                    tensor = model.board_to_tensor(board, current_player)
                else:
                    # 否則使用我們自定義的函數
                    tensor = board_to_tensor(board, current_player)
                
                # 檢查模型有哪些屬性和方法
                if hasattr(model, 'model') and callable(getattr(model, 'model', None)):
                    # 如果模型有 model 屬性且可調用
                    with torch.no_grad():
                        logits, _ = model.model(tensor)
                        policy = logits.view(BOARD_SIZE, BOARD_SIZE).cpu().numpy()
                elif callable(getattr(model, '__call__', None)):
                    # 如果模型本身可調用
                    with torch.no_grad():
                        logits, _ = model(tensor)
                        policy = logits.view(BOARD_SIZE, BOARD_SIZE).cpu().numpy()
                else:
                    # 嘗試通過索引或其他方式獲取策略
                    print("警告：模型沒有可調用的方法，使用隨機策略")
                    # 使用均勻隨機策略
                    policy = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32) / (BOARD_SIZE * BOARD_SIZE)

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
                
                # 儲存狀態
                if hasattr(model, 'board_to_tensor'):
                    state_tensor = model.board_to_tensor(board, current_player)
                    if isinstance(state_tensor, torch.Tensor):
                        state = state_tensor.squeeze(0).cpu().numpy()
                    else:
                        state = state_tensor  # 假設已經是 numpy 數組
                else:
                    state = board_to_tensor(board, current_player).squeeze(0).cpu().numpy()
                    
                states.append(state)
                policies.append(masked)
                players.append(current_player)

                flipped = apply_move_with_return(board, move, current_player)

                # 還棋邏輯：吃超過兩顆需還一顆
                if len(flipped) >= 2:
                    #print(f"還棋：玩家 {current_player} 吃了 {len(flipped)} 顆，返還 {flipped[0]}")
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
            
            # 顯示進度
            if (game_num + 1) % 5 == 0:
                print(f"已完成 {game_num + 1}/{NUM_SELFPLAY_GAMES} 局自我對弈")

        return data
        
    except Exception as e:
        print(f"生成自我對弈資料時出錯: {e}")
        traceback.print_exc()
        return []

def train_model(data, save_path="AlphaZero/model.pt"):
    
    try:
        # 嘗試創建模型
        model = AlphaZeroNet()
        print("訓練模型創建成功")
        
        # 檢查模型是否為 nn.Module
        if isinstance(model, nn.Module):
            model = model.to(device)
            print(f"模型已成功移至 {device} 裝置")
        else:
            print("警告: AlphaZeroNet 沒有正確繼承 nn.Module，繼續使用 CPU 訓練")
    
        # 無論是否已存在，先儲存目前尚未訓練的模型為初始版本
        if not os.path.exists("model_previous.pt"):
            try:
                if isinstance(model, nn.Module):
                    torch.save(model.state_dict(), "model_previous.pt")
                else:
                    # 假設模型有自定義的保存方法
                    model.save_model("AlphaZero/model_previous.pt")
                print("儲存初始（未訓練）模型為 model_previous.pt")
            except Exception as e:
                print(f"儲存初始模型失敗: {e}")
    
        # 如果已有訓練模型則載入
        if os.path.exists(save_path):
            try:
                if isinstance(model, nn.Module):
                    model.load_state_dict(torch.load(save_path, map_location=device))
                else:
                    # 假設模型有自定義的載入方法
                    model.load_model(save_path)
                print(f"已載入現有模型 {save_path}")
                
                # 儲存為過去模型
                if isinstance(model, nn.Module):
                    torch.save(model.state_dict(), "model_previous.pt")
                else:
                    model.save_model("AlphaZero/model_previous.pt")
            except Exception as e:
                print(f"載入模型失敗: {e}")
        
        # 檢查模型是否有參數可以優化
        if isinstance(model, nn.Module) and len(list(model.parameters())) > 0:
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # 加入 weight decay 防止過擬合
            
            # 交叉熵用於策略，MSE 用於值函數
            loss_fn_policy = nn.CrossEntropyLoss()
            loss_fn_value = nn.MSELoss()
    
            # 添加學習率調度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        else:
            print("警告: 模型沒有可訓練的參數或不是 nn.Module，將使用模型的自定義訓練方法")
            # 假設模型有自定義的訓練方法
            if hasattr(model, 'train_model'):
                model.train_model(data, save_path)
                return
            else:
                print("錯誤: 模型既不是 nn.Module 也沒有自定義的訓練方法")
                return
        
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
                    
                    # 修改部分: 將 policy 目標保留為原始形式，不轉換為索引
                    policy_targets = torch.from_numpy(np.array([x[1].flatten() for x in batch])).float().to(device)
                    value_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1).to(device)
    
                    # 預測
                    pred_policy, pred_value = model(state_batch)
                    
                    # 修改部分: 將預測的策略重塑為與目標相同的形狀
                    pred_policy = pred_policy.view(-1, 64)  # 重塑為 [batch_size, 64]
                    
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
                    traceback.print_exc()
                    continue
    
            # 計算平均損失
            avg_loss = total_loss / max(1, batch_count)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
            
            # 更新學習率
            scheduler.step(avg_loss)
            
            # 每個 epoch 結束後儲存模型
            try:
                
                save_path = "AlphaZero/model.pt"

                # 建立 models 資料夾（如果還沒存在）
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                if isinstance(model, nn.Module):
                    torch.save(model.state_dict(), save_path)  # 覆蓋主版本
                    torch.save(model.state_dict(), f"{save_path}.epoch{epoch+1}")
                else:
                    model.save_model(f"{save_path}.epoch{epoch+1}")
            except Exception as e:
                print(f"儲存 epoch {epoch+1} 模型失敗: {e}")
        
        # 儲存最終模型
        try:
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), save_path)
            else:
                model.save_model(save_path)
            print(f"模型已儲存至 {save_path}")
        except Exception as e:
            print(f"儲存最終模型失敗: {e}")
            
        print(f"成功處理的批次總數: {valid_batch_count}")
    
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        traceback.print_exc()

def evaluate_against_previous(new_model_path="model.pt", old_model_path="model_previous.pt", num_games=10):
    if not os.path.exists(old_model_path):
        print("找不到過去模型，無法進行對戰評估")
        return

    print("開始新舊模型對戰評估...")
    ai_new = AlphaZeroAI(model_path=new_model_path)
    ai_old = AlphaZeroAI(model_path=old_model_path)

    new_wins = 0
    old_wins = 0
    draws = 0

    for game_idx in range(num_games):
        board = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[3][3], board[4][4] = 2, 2
        board[3][4], board[4][3] = 1, 1
        current_player = 1
        pass_count = 0

        while pass_count < 2:
            legal_moves = get_legal_moves(board, current_player)
            if not legal_moves:
                pass_count += 1
                current_player = 3 - current_player
                continue

            pass_count = 0
            ai = ai_new if current_player == 1 else ai_old
            move = ai.choose_move(board, current_player, legal_moves)
            flipped = apply_move_with_return(board, move, current_player)

            if len(flipped) >= 2:
                return_one_piece(board, flipped[0], current_player)

            current_player = 3 - current_player

        black_count = sum(row.count(1) for row in board)
        white_count = sum(row.count(2) for row in board)
        if black_count > white_count:
            new_wins += 1
        elif black_count < white_count:
            old_wins += 1
        else:
            draws += 1

    print(f"對戰結果：新模型勝 {new_wins} 局，過去模型勝 {old_wins} 局，平局 {draws} 局")


if __name__ == "__main__":
    try:
        # 建立網絡模型用於自我對弈
        print("嘗試創建模型...")
        model = AlphaZeroNet()
        
        # 檢查模型是否正確繼承 nn.Module
        if not isinstance(model, nn.Module):
            print("警告: AlphaZeroNet 沒有正確繼承 nn.Module，無法使用 to() 方法")
            print("嘗試不使用 to() 方法繼續執行...")
        else:
            # 如果是 nn.Module 則可以使用 to() 方法
            model = model.to(device)
            print(f"模型已成功移至 {device} 裝置")
        
        # 如果模型存在則載入
        if os.path.exists("model.pt"):
            try:
                if isinstance(model, nn.Module):
                    model.load_state_dict(torch.load("model.pt", map_location=device))
                else:
                    # 如果不是 nn.Module，嘗試使用模型自定義的載入方法
                    model.load_model("model.pt")
                print("模型已成功載入")
            except Exception as e:
                print(f"載入模型失敗: {e}")
                print("使用未訓練模型繼續")
        else:
            print("尚未訓練模型，使用未訓練狀態")
        
        print("生成自我對弈資料中...")
        data = generate_selfplay_data(model)
        
        if data:
            print(f"生成了 {len(data)} 筆訓練資料")
            train_model(data, save_path="model.pt")
            evaluate_against_previous()
        else:
            print("未生成任何訓練資料，請檢查自我對弈邏輯")
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        print("錯誤詳情:")
        traceback.print_exc()