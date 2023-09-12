import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')

def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    print(N)
    n = np.arange(100)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X

# sampling rate
sr = 100
# sampling time
ts = 1.0/sr
# sampling frequency
fs = (2*np.pi)/ts 

t = np.arange(0,1,ts)

signal_freq = 5.
x = 3*np.sin(2*np.pi*signal_freq*t)

#freq = 4
#x += np.sin(2*np.pi*freq*t)

X = DFT(x)

# calculate the frequency
N = len(X)
print(N)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')
plt.show()

plt.figure(figsize = (8, 6))
plt.stem(freq, abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()

