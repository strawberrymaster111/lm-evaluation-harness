import matplotlib.pyplot as plt
import numpy as np

def plot_data(x, y, title="Sample Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig("sample_plot.png")


def test_plot_data():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    try:
        plot_data(x, y, title="Sine Wave", xlabel="Time", ylabel="Amplitude")
    except Exception as e:
        print(f"Plotting failed with exception: {e}")

if __name__ == "__main__":
    test_plot_data()
    print("Test passed!")