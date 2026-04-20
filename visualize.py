import matplotlib.pyplot as plt

def plot_dashboard():
    """Dummy visualization dashboard for pitch."""
    plt.figure(figsize=(10, 6))
    plt.title("Multi-Agent VSR-Env Dashboard")
    plt.plot([0, 50, 150, 200], [10, -50, 20, 100], label="Market Maker PnL")
    plt.plot([0, 50, 150, 200], [0, 60, -10, -80], label="Traders PnL")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (PnL)")
    plt.show()

if __name__ == "__main__":
    plot_dashboard()
