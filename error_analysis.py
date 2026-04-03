import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from config import OUTPUT_DIR

def plot_psd(epoch, sfreq=100):
    freqs, psd = welch(epoch, fs=sfreq)
    return freqs, psd


def main():
    X = np.load(OUTPUT_DIR / "X_epochs.npy")
    y_true = np.load(OUTPUT_DIR / "best_y_true.npy")
    y_pred = np.load(OUTPUT_DIR / "best_y_pred.npy")

    sfreq = 100

    n1_indices = np.where(y_true == 1)[0]

    correct = [i for i in n1_indices if y_pred[i] == 1][:5]
    wrong_w = [i for i in n1_indices if y_pred[i] == 0][:5]
    wrong_n2 = [i for i in n1_indices if y_pred[i] == 2][:5]

    def plot_group(indices, title, filename):
        plt.figure()
        for idx in indices:
            freqs, psd = plot_psd(X[idx], sfreq)
            plt.plot(freqs, psd, alpha=0.5)
        plt.title(title)
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()

    plot_group(correct, "Correct N1 PSD", "n1_correct_psd.png")
    plot_group(wrong_w, "N1 misclassified as Wake", "n1_to_w_psd.png")
    plot_group(wrong_n2, "N1 misclassified as N2", "n1_to_n2_psd.png")


if __name__ == "__main__":
    main()