import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

file_path = 'p_diff_avg.txt'
# x_vals = []
# y_vals = []

#index = count()
def animate(i):
    data = pd.read_csv(file_path)
    #x = data['x_value']
    #y1 = data['total_1']
    #y2 = data['total_2']

    plt.cla()

    plt.plot(data, label='Channel 1')
    #plt.plot(x, y2, label='Channel 2')

    plt.legend(loc='upper left')
    plt.tight_layout()


#data = pd.read_csv(file_path)
ani = FuncAnimation(plt.gcf(), animate, interval=100)

plt.tight_layout()
plt.show()
