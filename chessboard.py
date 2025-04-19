import tkinter as tk
from tkinter import messagebox
import string, json, os

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
        self.setup_score_panel(self.left_panel, is_player1=True)

        # 棋盤
        self.canvas = tk.Canvas(self.main_frame, width=400, height=400, bg="#4CAF50", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10)
        self.cell_size = 50

        # 右記分板
        self.right_panel = tk.Frame(self.main_frame, bg="#3B3B3B", width=120)
        self.right_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.setup_score_panel(self.right_panel, is_player1=False)

        # 底部控制區
        self.bottom_frame = tk.Frame(master, bg="#2E2E2E")
        self.bottom_frame.pack(pady=5)

        self.first_var = tk.IntVar(value=1)
        tk.Radiobutton(self.bottom_frame, text="玩家1先手", variable=self.first_var, value=1, bg="#2E2E2E", fg="white", selectcolor="#4CAF50").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(self.bottom_frame, text="玩家2先手", variable=self.first_var, value=2, bg="#2E2E2E", fg="white", selectcolor="#4CAF50").pack(side=tk.LEFT, padx=10)
        tk.Button(self.bottom_frame, text="開始遊戲", command=self.start_game, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(self.bottom_frame, text="重新開始", command=self.reset_board, bg="#F44336", fg="white").pack(side=tk.LEFT, padx=10)

        # 狀態列
        self.status = tk.Label(master, text="尚未開始", anchor=tk.W, bg="#2E2E2E", fg="white")
        self.status.pack(fill=tk.X, padx=10, pady=5)

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
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="#2e2e2e", fill="#3f7f5f")

    def draw_piece(self, row, col, color):
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        if color == "black":
            self.canvas.create_image(x, y, image=self.black_img)
        elif color == "white":
            self.canvas.create_image(x, y, image=self.white_img)

    def init_pieces(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.board[3][3] = 2
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[4][4] = 2
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
        self.init_pieces()
        self.redraw_pieces()
        name = self.p1_name_var.get() if self.current_player == 1 else self.p2_name_var.get()
        color = "黑" if self.current_player == 1 else "白"
        self.status.config(text=f"{name} ({color}) 開始下子")
        self.canvas.bind("<Button-1>", self.on_click)

    def reset_board(self):
        if messagebox.askyesno("重新開始", "確定要重新開始？"):
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
        for dx, dy in DIRECTIONS:
            x, y = row + dx, col + dy
            flip = []
            while 0 <= x < 8 and 0 <= y < 8 and self.board[x][y] == opponent:
                flip.append((x, y))
                x += dx
                y += dy
            if flip and 0 <= x < 8 and 0 <= y < 8 and self.board[x][y] == player:
                for fx, fy in flip:
                    self.board[fx][fy] = player

    def on_click(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        if not self.is_valid_move(row, col, self.current_player):
            return

        self.board[row][col] = self.current_player
        self.flip_pieces(row, col, self.current_player)

        next_player = 2 if self.current_player == 1 else 1

        if self.get_valid_moves(next_player):
            self.current_player = next_player
            self.redraw_pieces()
            name = self.p1_name_var.get() if self.current_player == 1 else self.p2_name_var.get()
            ctext = "黑" if self.current_player == 1 else "白"
            self.status.config(text=f"{name} ({ctext}) 的回合")

        elif self.get_valid_moves(self.current_player):
            self.redraw_pieces()
            self.status.config(text=f"{'黑方' if self.current_player == 1 else '白方'} 無法落子，PASS！")

        else:
            self.redraw_pieces()
            self.status.config(text="遊戲結束！")
            self.canvas.unbind("<Button-1>")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReversiGUI(root)
    root.mainloop()
