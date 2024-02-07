import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
import time
plt.style.use('fivethirtyeight')

# x_vals = []
# y_vals = []

#index = count()
file_path = 'p_diff_avg.txt'
#SAMPLE_RATE = 44100  # Hertz
#DURATION = 5  # Seconds


def animate(i):
    start_time = time.time()

    data = pd.read_csv(file_path)
    # x = data['x_value']
    # y1 = data['total_1']
    # y2 = data['total_2']

    plt.cla()

    plt.plot(data[-1000:-1], label='Channel pdiffAvg')
    plt.ylim(bottom=-1050, top=1050)  # Set the y-axis limits
    #plt.text(0, 0, f"data {data[-1000:-1]}", fontsize=12)

    #plt.plot(x, y2, label='Channel 2')
 
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time for bandpass filter:", execution_time, "seconds")

    #plt.gca().cla()


# ani = FuncAnimation(plt.gcf(), animate, interval=100)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time for bandpass filter:", execution_time, "seconds")

ani = FuncAnimation(plt.gcf(), animate, interval=5)

plt.tight_layout()
plt.show()