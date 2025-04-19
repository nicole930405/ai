import tkinter as tk
from tkinter import messagebox
import string, json, os
import random
import time

STAT_FILE = "player_stats.json"

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),         ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

class ReversiGUI:
    def __init__(self, master):
        self.master = master
        master.title("黑白棋")
        master.configure(bg="#ADB5AB")

        self.black_img = tk.PhotoImage(file="black.png")
        self.white_img = tk.PhotoImage(file="white.png")

        # 讀取或初始化玩家統計
        if os.path.exists(STAT_FILE):
            with open(STAT_FILE, "r", encoding="utf-8") as f:
                self.stats = json.load(f)
        else:
            self.stats = {}

        # 主框架
        self.main_frame = tk.Frame(master, bg="#2E2E2E")
        self.main_frame.pack(padx=10, pady=10)

        # 左記分板
        self.left_panel = tk.Frame(self.main_frame, bg="#3B3B3B", width=120)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.info_label_r = tk.Label(self.left_panel, 
                                     text="注意事項： \n棋盤底色變深 -> 被吃掉\n棋盤底色變淺 -> 還回去", 
                                     bg="#3B3B3B", 
                                     fg="white", 
                                     font=("Arial", 10), 
                                     justify="left",
                                     anchor="nw", 
                                     wraplength=200, 
                                     )
        self.info_label_r.place(x=0, y=0, width=200, height=60)
        self.setup_score_panel(self.left_panel, is_player1=True)

        # 棋盤
        self.canvas = tk.Canvas(self.main_frame, width=400, height=400, bg="#4CAF50", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10)
        self.cell_size = 50

        # 右記分板
        self.right_panel = tk.Frame(self.main_frame, bg="#3B3B3B", width=120)
        self.right_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        # 放在右側記分板上方的提示文字框
        self.info_label = tk.Label(self.right_panel, text="", bg="#3B3B3B", fg="white", font=("Arial", 10), justify="left",anchor="nw", wraplength=200)
        self.info_label.place(x=0, y=0, width=200, height=60)
        self.setup_score_panel(self.right_panel, is_player1=False)

        # 底部控制區
        self.bottom_frame = tk.Frame(master, bg="#2E2E2E")
        self.bottom_frame.pack(pady=5)

        self.first_var = tk.IntVar(value=1)
        self.computer_player = 2
        tk.Radiobutton(self.bottom_frame, text="玩家1先手", variable=self.first_var, value=1, bg="#2E2E2E", fg="white", selectcolor="#4CAF50").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(self.bottom_frame, text="玩家2先手", variable=self.first_var, value=2, bg="#2E2E2E", fg="white", selectcolor="#4CAF50").pack(side=tk.LEFT, padx=10)
        tk.Button(self.bottom_frame, text="開始遊戲", command=self.start_game, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(self.bottom_frame, text="重新開始", command=self.reset_board, bg="#F44336", fg="white").pack(side=tk.LEFT, padx=10)

        # 狀態列
        self.status = tk.Label(master, text="尚未開始", anchor=tk.W, bg="#2E2E2E", fg="white")
        self.status.pack(fill=tk.X, padx=10, pady=5)
        self.last_flipped_positions = []# 記住上回合吃掉的座標
        self.last_returned_position = None

        self.total_time = 0
        self.start_time = None
        self.time_label = tk.Label(self.bottom_frame, text="每一步時間：0.00 秒 | 總時間：0.00 秒", bg="#2E2E2E", fg="white")
        self.time_label.pack(side=tk.LEFT, padx=10)

        self.draw_board()

    def setup_score_panel(self, panel, is_player1):
        name = tk.StringVar()
        container = tk.Frame(panel, bg=panel["bg"])
        container.pack(expand=True)

        player_label = tk.Label(container,
                            text="玩家 1" if is_player1 else "玩家 2",
                            font=("Arial", 12, "bold"),
                            bg=panel["bg"], fg="white")
        player_label.pack(pady=(20, 5)) 

        entry = tk.Entry(container, textvariable=name, font=("Arial", 12, "bold"), justify="center")
        entry.pack(pady=(20, 10), fill=tk.X, padx=10)

        save_btn = tk.Button(container, text="儲存名稱", command=self.save_names, bg="#4CAF50", fg="white")
        save_btn.pack(pady=(0, 10))

        score_label = tk.Label(container, text="分數：0", font=("Arial", 12), bg=panel["bg"], fg="white")
        score_label.pack()

        if is_player1:
            self.p1_name_var = name
            self.p1_score_label = score_label
        else:
            self.p2_name_var = name
            self.p2_score_label = score_label

        entry.insert(0, "黑方" if is_player1 else "白方")

    def draw_board(self):
        self.canvas.delete("all")
        for row in range(8):
            for col in range(8):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                # 若這格剛剛被翻 → 顯示深色
                pos = (row, col)
                if pos == getattr(self, "last_returned_position", None):
                    color = "#5f9f7f"
                elif pos in getattr(self, 'last_flipped_positions', []):
                    color = "#2e5f3f"  # 深綠
                else:
                    color = "#3f7f5f"  # 原綠

                self.canvas.create_rectangle(x0, y0, x1, y1, outline="#2e2e2e", fill=color)

    def draw_piece(self, row, col, color):
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        if color == "black":
            self.canvas.create_image(x, y, image=self.black_img)
        elif color == "white":
            self.canvas.create_image(x, y, image=self.white_img)

    def init_pieces(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.board[3][3] = 1
        self.board[3][4] = 2
        self.board[4][3] = 2
        self.board[4][4] = 1
        self.redraw_pieces()

    def redraw_pieces(self):
        self.draw_board()
        black = white = 0
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 1:
                    self.draw_piece(row, col, "black")
                    black += 1
                elif self.board[row][col] == 2:
                    self.draw_piece(row, col, "white")
                    white += 1

        # 顯示可落子位置（黃色圓點）
        
        for row, col in self.get_valid_moves(self.current_player):
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="yellow", outline="")

        self.p1_score_label.config(text=f"分數：{black}")
        self.p2_score_label.config(text=f"分數：{white}")

    def start_game(self):
        self.current_player = self.first_var.get()
        self.computer_player = 2 if self.current_player == 1 else 1
        self.init_pieces()
        self.redraw_pieces()
        self.total_time = 0
        self.start_time = time.time()
        self.update_time_label(0)
        
        if self.current_player == self.computer_player:
            self.status.config(text="電腦思考中...")
            self.master.after(500, self.computer_move)
        else:
            name = self.p1_name_var.get()
            color = "黑" if self.current_player == 1 else "白"
            self.status.config(text=f"{name} ({color}) 開始下子")
        
        self.canvas.bind("<Button-1>", self.on_click)

    def reset_board(self):
        if messagebox.askyesno("重新開始", "確定要重新開始？"):
            self.last_flipped_positions = []
            self.last_returned_position = None
            self.info_label.config(text="")  # 清空右側提示欄
            self.canvas.unbind("<Button-1>")
            self.draw_board()
            self.status.config(text="已重置，請按「開始遊戲」")
            self.board = [[0 for _ in range(8)] for _ in range(8)]
            self.p1_score_label.config(text="分數：0")
            self.p2_score_label.config(text="分數：0")

    def save_names(self):
        for pname in (self.p1_name_var.get(), self.p2_name_var.get()):
            if pname not in self.stats:
                self.stats[pname] = {"wins": 0, "losses": 0, "draws": 0}
        with open(STAT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("儲存成功", "玩家名稱已儲存！")

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != 0:
            return False
        opponent = 2 if player == 1 else 1
        for dx, dy in DIRECTIONS:
            x, y = row + dx, col + dy
            count = 0
            while 0 <= x < 8 and 0 <= y < 8 and self.board[x][y] == opponent:
                x += dx
                y += dy
                count += 1
            if count > 0 and 0 <= x < 8 and 0 <= y < 8 and self.board[x][y] == player:
                return True
        return False

    def get_valid_moves(self, player):
        return [(r, c) for r in range(8) for c in range(8) if self.is_valid_move(r, c, player)]

    def flip_pieces(self, row, col, player):
        opponent = 2 if player == 1 else 1
        total_flips = [] # 收集所有可翻轉的棋子座標
        flip_info_by_direction = []
        for dx, dy in DIRECTIONS:
            x, y = row + dx, col + dy
            flip = []
            while 0 <= x < 8 and 0 <= y < 8 and self.board[x][y] == opponent:
                flip.append((x, y))
                x += dx
                y += dy
            if flip and 0 <= x < 8 and 0 <= y < 8 and self.board[x][y] == player:
                total_flips.extend(flip) # 加入所有方向的可翻轉棋子
                flip_info_by_direction.append(flip)

        removed = None
        original_count = len(total_flips)
        if len(total_flips) >2:
            removed = random.choice(total_flips)
            total_flips.remove(removed)
            self.last_returned_position = removed
        else:
            self.last_returned_position = None

        self.last_flipped_positions = total_flips.copy()

        
        for fx, fy, in total_flips:
            self.board[fx][fy] = player

        # 組合訊息
        symbol = "黑" if player == 1 else "白"
        other = "白" if player == 1 else "黑"
        eat_info = "".join([f"({fx},{fy})" for fx, fy in total_flips])
        return_info = f" 還{removed}給{other}" if removed else ""
        if original_count == 1:
            msg = f"{symbol}棋：({row},{col})只吃對方一棋子! {other}棋：{eat_info} 被吃 \n {return_info}"
        elif original_count > 2:
            msg = f"{symbol}棋：({row},{col})只吃對方一棋子! {other}棋：{eat_info} 被吃 \n {return_info}"
        else:
            msg = "None"
        return msg

    def draw_info_text(self, message, color = "white"):
        # full_message = "（此提示將於 5 秒後自動消失）\n"+ message 
        self.info_label.config(text=message, fg = color)
        # self.master.after(5000, lambda: self.info_label.config(text=""))

    def on_click(self, event):

        if self.current_player == self.computer_player:
            return  # 如果當前是電腦，不接受點擊
        
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        if not self.is_valid_move(row, col, self.current_player):
            return
        
        end_time = time.time()
        step_time = end_time - self.start_time
        self.total_time += step_time
        self.update_time_label(step_time)
        
        self.last_flipped_positions = []  # 清除上次被吃提示
        self.board[row][col] = self.current_player
        #先翻棋取得訊息
        flip_message = self.flip_pieces(row, col, self.current_player)
        self.redraw_pieces()

        if flip_message != "None":
            self.draw_info_text(flip_message)

        self.start_time = time.time()
        self.switch_player()

    def switch_player(self):
        self.current_player = 2 if self.current_player == 1 else 1

        if self.get_valid_moves(self.current_player):
            self.redraw_pieces()
            if self.current_player == self.computer_player:
                self.status.config(text="電腦思考中...")
                self.master.after(500, self.computer_move)
            else:
                name = self.p1_name_var.get()
                color = "黑" if self.current_player == 1 else "白"
                self.status.config(text=f"{name} ({color}) 的回合")
        else:
            if self.get_valid_moves(1 if self.current_player == 2 else 2):
                self.status.config(text=f"{'電腦' if self.current_player == self.computer_player else '玩家'}無法落子，PASS！")
                self.draw_info_text("PASS!", color="red")
                self.current_player = 1 if self.current_player == 2 else 2
                if self.current_player == self.computer_player:
                    self.master.after(500, self.computer_move)
            else:
                self.end_game()

    def computer_move(self):
        step_start_time = time.time()
        valid_moves = self.get_valid_moves(self.computer_player)
        if not valid_moves:
            self.status.config(text="電腦無法落子，PASS！")
            self.draw_info_text("電腦無法落子，PASS！", color="red")
            self.current_player = 1 if self.computer_player == 2 else 2
            self.start_time = time.time()
            return
        
        def make_move():
            row, col = random.choice(valid_moves)
            self.board[row][col] = self.computer_player
            flip_message = self.flip_pieces(row, col, self.computer_player)
            self.redraw_pieces()

            if flip_message != "None":
                self.draw_info_text(flip_message)

            step_time = time.time() - step_start_time
            self.total_time += step_time
            self.update_time_label(step_time)

            self.start_time = time.time()
            self.switch_player()

        # 執行延遲後再記錄時間
        self.master.after(500, make_move)


    def update_time_label(self, step_time):
        self.time_label.config(text=f"每一步時間：{step_time:.2f} 秒 | 總時間：{self.total_time:.2f} 秒")

    

                

if __name__ == "__main__":
    root = tk.Tk()
    app = ReversiGUI(root)
    root.mainloop()
