import matplotlib.pyplot as plt
import string

def draw_reversi_board():
    fig, ax = plt.subplots(figsize=(6, 6))
    for row in range(8):         
        for col in range(8):     
            # 綠底方格
            rect = plt.Rectangle(
                (col, row), 1, 1,
                facecolor='green',
                edgecolor='black'
            )
            ax.add_patch(rect)
            # 座標標示（列 A~H，行 1~8）
            coord = f"{string.ascii_uppercase[col]}{row + 1}"
            ax.text(
                col + 0.5, row + 0.5, coord,
                ha='center', va='center',
                color='white', fontsize=12
            )
    # 設定長寬比
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.title("黑白棋棋盤")
    plt.show()

if __name__ == "__main__":
    draw_reversi_board()
