import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import freqs
import scipy.io

# Function that applies butterwoth filter
def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Calculate the harmonic Difference
def calc_thd(signal, sample_rate, fundamental_frequency):
    # Perform the FFT on the signal    
    fft = np.abs(np.fft.fft(signal))
    # # Get the frequencies associated with the FFT    
    frequencies = np.fft.fftfreq(len(signal), d=1/sample_rate) 
    # # Get the index of the fundamental frequency component    
    index = np.argmin(np.abs(frequencies - fundamental_frequency))
    # # Get the magnitude of the fundamental frequency component    
    fundamental = fft[index]
    # Calculate the sum of the magnitudes of all the harmonic components    
    harmonic_sum = np.sum(fft[(frequencies > 0) & (frequencies != fundamental_frequency)]) 
    # # Calculate the THD as the ratio of the sum of the harmonic components to the magnitude of the fundamental component    
    thd = harmonic_sum / fundamental
    return thd

# remove mean from the signal
def remove_mean(signal):
    mean= np.mean(signal)
    filtered_data=[]
    for point in signal:
        if point>signal:
            filtered_data.append(point)

    return filtered_data


# Define the input sine function (for testing)
def input_sine(f, t):
    return np.sin(2 * np.pi * f * t)

# Define the sampling rate and time range
sampling_rate = 100
time = np.linspace(0, 1, sampling_rate)
frequency = 5

# Define the frequency of the input sine function
mat = scipy.io.loadmat('100m.mat')
val= mat["val"]
val=val[0]
val_butter= butter_lowpass_filter(val, 60, 1000, 3)

#filtered_data = remove_mean(val_butter)
plt.plot(val,'r')
plt.plot(val_butter)
plt.show()

# Get the input signal by evaluating the sine function over the time range
input_signal = input_sine(frequency, time)
#input_signal = val_butter

# Perform the DFT
"""
def DFT(x):
    N = len(x)
    print(N)
    n = np.arange(100)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)   
    X = np.dot(e, x)
    return X
"""
dft = np.fft.fft(input_signal)

# Get the magnitude of the DFT
dft_magnitude = np.abs(dft)

# Get the frequency Dictionary 
freqs = np.fft.fftfreq(len(input_signal))
print(freqs)

#for coef, freq in zip(dft, freqs):
#    if coef:
#        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef, f=freq))

harmonic_distortion=calc_thd(input_signal, 100, 10000)
print(harmonic_distortion)

#Plot the input signal and the DFT magnitude
plt.figure(figsize = (8, 6))
plt.stem(input_signal)
plt.ylabel('Amplitude')
plt.show()

plt.stem(dft_magnitude)
plt.show()