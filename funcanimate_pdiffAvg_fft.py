import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
import numpy as np
plt.style.use('fivethirtyeight')

# x_vals = []
# y_vals = []

#index = count()
file_path = 'p_diff_avg.txt'
SAMPLE_RATE = 20  # Hertz
DURATION = 10  # Seconds
N = SAMPLE_RATE * DURATION


def animate(i):
    #N = 100
    data = pd.read_csv(file_path)
    data = data.iloc[:, 0]
    fdata = data[-1*(N+1):-1]
    # x = data['x_value']
    # y1 = data['total_1']
    # y2 = data['total_2']
        
    yf = fft(fdata.values)
    xf = fftfreq(N, 1 / SAMPLE_RATE)
    print("yf",np.abs(yf))
    #print("xf",xf)
    plt.cla()
    
    plt.plot(xf, np.abs(yf), label='Channel 1')
    #plt.show()
    

    #plt.plot(data[-1000:-1], label='Channel 1')
    #plt.plot(x, y2, label='Channel 2')

    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.gca().cla()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()