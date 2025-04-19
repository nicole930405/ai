import tkinter as tk
from tkinter import messagebox
import string, json, os

STAT_FILE = "player_stats.json"

class ReversiGUI:
    def __init__(self, master):
        self.master = master
        master.title("黑白棋")

        # 讀取或初始化玩家統計
        if os.path.exists(STAT_FILE):
            with open(STAT_FILE, "r", encoding="utf-8") as f:
                self.stats = json.load(f)
        else:
            self.stats = {}

        # ===== 上方控制區域 =====
        ctrl = tk.Frame(master)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(ctrl, text="玩家 1 (黑)：").grid(row=0, column=0)
        self.p1_name = tk.Entry(ctrl, width=10)
        self.p1_name.insert(0, "黑方")
        self.p1_name.grid(row=0, column=1)

        tk.Label(ctrl, text="玩家 2 (白)：").grid(row=0, column=2)
        self.p2_name = tk.Entry(ctrl, width=10)
        self.p2_name.insert(0, "白方")
        self.p2_name.grid(row=0, column=3)

        # 先後手選擇
        self.first_var = tk.IntVar(value=1)
        tk.Radiobutton(ctrl, text="玩家1先手", variable=self.first_var, value=1).grid(row=0, column=4)
        tk.Radiobutton(ctrl, text="玩家2先手", variable=self.first_var, value=2).grid(row=0, column=5)

        tk.Button(ctrl, text="開始遊戲", command=self.start_game).grid(row=0, column=6, padx=5)
        tk.Button(ctrl, text="重新開始", command=self.reset_board).grid(row=0, column=7)
        tk.Button(ctrl, text="儲存名稱", command=self.save_names).grid(row=0, column=8)

        # ===== 棋盤區域 =====
        self.canvas = tk.Canvas(master, width=400, height=400, bg="green")
        self.canvas.pack(padx=5, pady=5)
        self.cell_size = 50  
        self.draw_board()

        # ===== 底部狀態列 =====
        self.status = tk.Label(master, text="尚未開始", anchor=tk.W)
        self.status.pack(fill=tk.X, padx=5, pady=5)

    def draw_board(self):
        """畫出 8×8 綠底方格並標示座標（A1 at top-left, 由上往下遞增）"""
        self.canvas.delete("all")
        for row in range(8):
            for col in range(8):
                x0 = col * self.cell_size
                y0 = row * self.cell_size      # row=0 at top
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                # 畫格子
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill="green")
                # 標示座標：列 A~H，行 1~8
                coord = f"{string.ascii_uppercase[col]}{row+1}"
                self.canvas.create_text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    text=coord,
                    fill="white",
                    font=("Arial", 12)
                )

    def start_game(self):
        """開始遊戲，顯示先手玩家"""
        self.current_player = self.first_var.get()
        name = self.p1_name.get() if self.current_player == 1 else self.p2_name.get()
        color = "黑" if self.current_player == 1 else "白"
        self.status.config(text=f"{name} ({color}) 開始下子")
        # TODO: 在此初始化棋子並綁定點擊事件 self.canvas.bind("<Button-1>", self.on_click)

    def reset_board(self):
        """重新開始一局"""
        if messagebox.askyesno("重新開始", "確定要重新開始？"):
            self.draw_board()
            self.status.config(text="已重置，請按「開始遊戲」")

    def save_names(self):
        """儲存玩家名稱至統計檔案"""
        for pname in (self.p1_name.get(), self.p2_name.get()):
            if pname not in self.stats:
                self.stats[pname] = {"wins": 0, "losses": 0, "draws": 0}
        with open(STAT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("儲存成功", "玩家名稱已儲存！")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReversiGUI(root)
    root.mainloop()
