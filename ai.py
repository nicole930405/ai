# ai.py
import copy, math
from hello import DIRECTIONS    # ← 這裡改成 hello 而不是 reversi

def evaluate(board, player):
    my, opp = player, 2 if player==1 else 1
    return sum(row.count(my) for row in board) - sum(row.count(opp) for row in board)

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

def minimax(board, player, depth, α, β, maximizing):
    moves = get_valid_moves(board, player if maximizing else (2 if player==1 else 1))
    if depth==0 or not moves:
        return evaluate(board, player), None

    best_move = None
    if maximizing:
        value = -math.inf
        for m in moves:
            nb = apply_move(board, m, player)
            v,_ = minimax(nb, player, depth-1, α, β, False)
            if v>value: value, best_move = v, m
            α = max(α, value)
            if α>=β: break
        return value, best_move
    else:
        value = math.inf
        opp = 2 if player==1 else 1
        for m in moves:
            nb = apply_move(board, m, opp)
            v,_ = minimax(nb, player, depth-1, α, β, True)
            if v<value: value, best_move = v, m
            β = min(β, value)
            if α>=β: break
        return value, best_move

def get_best_move(board, player, max_depth=4):
    _, move = minimax(board, player, max_depth, -math.inf, math.inf, True)
    return move
