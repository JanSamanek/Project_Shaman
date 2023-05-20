import matplotlib.pyplot as plt

def plot_and_save_kalman_data(data1, data2, data3):
    x1, y1 = zip(*data1)
    x2, y2 = zip(*data2)
    x3, y3 = zip(*data3)
    fig, ax = plt.subplots()
    ax.plot(x1, y1, 'r-', label='measured position')
    ax.plot(x2, y2, 'b-', label='filtered postition')
    ax.plot(x3, y3, 'g-', label='predicted position')

    ax.set_xlabel('x position ')
    ax.set_ylabel('y position')
    ax.set_title('Kalman filtr data')
    ax.legend()
    plt.savefig('kalman_plot.png')
