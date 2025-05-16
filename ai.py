# ai.py
import copy, math
from hello import DIRECTIONS    # ← 這裡改成 hello 而不是 reversi
import time

import numpy as np
from tensorflow.keras.models import load_model

model = load_model("reversi_policy_model.h5")

def predict_by_model(board, valid_moves):
    input_board = np.array(board).reshape(1, 8, 8, 1).astype(np.float32) / 2.0
    prediction = model.predict(input_board, verbose=0)[0].reshape(8, 8)

    # 只考慮合法的落子點，選機率最大的那一格
    best_move = None
    best_score = -float("inf")
    for r, c in valid_moves:
        if prediction[r][c] > best_score:
            best_score = prediction[r][c]
            best_move = (r, c)
    return best_move

POSITION_WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

X_SQUARES = [
    (0,1), (1,0), (1,1),
    (0,6), (1,7), (1,6),
    (6,0), (7,1), (6,1),
    (7,6), (6,7), (6,6)
]



def is_stable(board, r, c):
    player = board[r][c]
    if player == 0:
        return False

    # 角落 → 永遠穩定
    if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
        return True

    # 若該位置在邊界，且一整列/行都是我方子 → 穩定
    if r in [0, 7] and all(board[r][i] == player for i in range(8)):
        return True
    if c in [0, 7] and all(board[i][c] == player for i in range(8)):
        return True

    # 可擴充更多判定（如邊角展延穩定區），但這樣已足以增強 AI 水準
    return False



def evaluate(board, player):
    opp = 2 if player == 1 else 1
    my_score = opp_score = 0
    my_moves = len(get_valid_moves(board, player))
    opp_moves = len(get_valid_moves(board, opp))
    my_stable = opp_stable = 0
    my_discs = opp_discs = 0
    x_penalty = 0
    flip_advantage = 0

    total_discs = 0
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece != 0:
                total_discs += 1

            if piece == player:
                my_discs += 1
                my_score += POSITION_WEIGHTS[r][c]
                if is_stable(board, r, c):
                    my_stable += 1
                if (r, c) in X_SQUARES:
                    x_penalty -= 15
            elif piece == opp:
                opp_discs += 1
                opp_score += POSITION_WEIGHTS[r][c]
                if is_stable(board, r, c):
                    opp_stable += 1
                if (r, c) in X_SQUARES:
                    x_penalty += 15

    # 1️⃣ 穩定子
    stability_score = 15 * (my_stable - opp_stable)

    # 2️⃣ 行動力
    mobility_score = 0
    if my_moves + opp_moves > 0:
        mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

    # 3️⃣ 吃子數量（中期重要）
    # → 假設電腦每回合都吃比對手多的，這是好事
    flip_score = (my_discs - opp_discs) * 2 if total_discs < 54 else 0  # 避免終局反效果

    # 4️⃣ 終局以子數勝負
    endgame_score = 0
    if total_discs >= 58:
        endgame_score = 500 * (my_discs - opp_discs) / (my_discs + opp_discs)

    return (my_score - opp_score) + stability_score + mobility_score + x_penalty + flip_score + endgame_score

def get_valid_moves(board, player):
    opp = 2 if player==1 else 1
    moves = []
    for r in range(8):
        for c in range(8):
            if board[r][c]!=0: continue
            for dx,dy in DIRECTIONS:
                x,y = r+dx, c+dy; cnt=0
                while 0<=x<8 and 0<=y<8 and board[x][y]==opp:
                    x+=dx; y+=dy; cnt+=1
                if cnt>0 and 0<=x<8 and 0<=y<8 and board[x][y]==player:
                    moves.append((r,c))
                    break
    return moves

def apply_move(board, move, player):
    newb = copy.deepcopy(board)
    r,c = move; newb[r][c]=player
    opp = 2 if player==1 else 1
    for dx,dy in DIRECTIONS:
        x,y = r+dx, c+dy; flip=[]
        while 0<=x<8 and 0<=y<8 and newb[x][y]==opp:
            flip.append((x,y)); x+=dx; y+=dy
        if flip and 0<=x<8 and 0<=y<8 and newb[x][y]==player:
            for fx,fy in flip: newb[fx][fy]=player
    return newb

def minimax(board, player, depth, alpha, beta, maximizing, start_time, time_limit):
    """
    Alpha-Beta 搭配 Move Ordering + Early Cutoff（時間限制 & 終局優化）
    """
    # ⏰ 檢查時間是否超過限制
    if time.time() - start_time > time_limit:
        return evaluate(board, player), None

    # 🏁 若棋局接近結束，直接評估盤面（不要再展開）
    if sum(row.count(0) for row in board) <= 6:
        return evaluate(board, player), None

    current_player = player if maximizing else (2 if player == 1 else 1)
    moves = get_valid_moves(board, current_player)

    if depth == 0 or not moves:
        return evaluate(board, player), None

    # 🎯 Move Ordering：優先角落、邊緣（剪枝更快）
    moves.sort(key=lambda m: POSITION_WEIGHTS[m[0]][m[1]], reverse=maximizing)

    best_move = None

    if maximizing:
        value = -math.inf
        for move in moves:
            new_board = apply_move(board, move, player)
            v, _ = minimax(new_board, player, depth - 1, alpha, beta, False, start_time, time_limit)
            if v > value:
                value = v
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        opp = 2 if player == 1 else 1
        for move in moves:
            new_board = apply_move(board, move, opp)
            v, _ = minimax(new_board, player, depth - 1, alpha, beta, True, start_time, time_limit)
            if v < value:
                value = v
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move
    
# def get_best_move(board, player, max_depth=2, time_limit=0.01):

#     """
#     使用 Iterative Deepening + Move Ordering，找出最佳下一步。
#     若超過 time_limit（秒），會提早回傳上一輪結果。
#     """
#     start_time = time.time()
#     best_move = None
#     current_depth = 1

#     while current_depth <= max_depth:
#         value, move = minimax(board, player, current_depth, -math.inf, math.inf, True, start_time, time_limit)
#         if time.time() - start_time > time_limit:
#             break
#         best_move = move
#         current_depth += 1

#     return best_move

def get_best_move(board, player, max_depth=2, time_limit=0.05, use_model=False):
    valid_moves = get_valid_moves(board, player)
    if not valid_moves:
        return None

    if use_model:
        return predict_by_model(board, valid_moves)
    else:
        # 原本的 Alpha-Beta 策略
        start_time = time.time()
        best_move = None
        current_depth = 1
        while current_depth <= max_depth:
            value, move = minimax(board, player, current_depth, -math.inf, math.inf, True, start_time, time_limit)
            if time.time() - start_time > time_limit:
                break
            best_move = move
            current_depth += 1
        return best_move

def choose_best_return_piece(board, pending_return, player):
    """
    從 pending_return 裡選出一個「還回去對對手最沒利」的點。
    """
    import copy

    best_score = -math.inf
    best_piece = None
    opp = 2 if player == 1 else 1

    for rx, ry in pending_return:
        # 模擬一個 board：還這顆給對手，其它還是自己顏色
        test_board = copy.deepcopy(board)
        for fx, fy in pending_return:
            test_board[fx][fy] = player
        test_board[rx][ry] = opp  # 還這一顆

        score = evaluate(test_board, player)
        if score > best_score:
            best_score = score
            best_piece = (rx, ry)

    return best_piece

