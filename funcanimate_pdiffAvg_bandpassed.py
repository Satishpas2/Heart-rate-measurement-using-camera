import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import butter, lfilter
import time
plt.style.use('fivethirtyeight')

# x_vals = []
# y_vals = []

#index = count()
file_path = 'p_diff_avg.txt'
SAMPLE_RATE = 20  # Hertz
#DURATION = 10  # Seconds
#N = SAMPLE_RATE * DURATION

# Sample rate and desired cutoff frequencies (in Hz).
fs = SAMPLE_RATE
lowcut = .2
highcut = .5
x_axis = 0
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def animate(i):
    #global x_axis  # Declare x_axis as global

    start_time = time.time()

    data = pd.read_csv(file_path)
    data = data.iloc[:, 0]
    fdata =data[-1000:-1]
    plt.cla()

    yf = butter_bandpass_filter(fdata, lowcut, highcut, fs, order=6)

    yf_df = pd.DataFrame(yf, columns=['Channel pdiffAvg_BandPassed'])
    yf_df.index = fdata.index
    #yf_df.index = pd.RangeIndex(start=x_axis, stop=x_axis+999, step=1)
    #x_axis = x_axis + 999
    plt.plot(yf_df, label='Channel pdiffAvg_BandPassed')
    plt.ylim(bottom=-550, top=550)  # Set the y-axis limits
    plt.legend(loc='upper left')
    plt.tight_layout()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time for bandpass filter:", execution_time, "seconds")

ani = FuncAnimation(plt.gcf(), animate, interval=5)

plt.tight_layout()
plt.show()