from scipy.fft import fft
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def fourier_transform(wave):
    '''
    Derive frequency spectrum of a signal from time domain
    :param wave: signal in the time domain
    :returns frequencies and their content distribution
    '''

    # zero-centering
    average = np.average(wave)
    wave = wave - average

    magnitude = fft(wave)# Fourier computing

    return magnitude, average
    

'''
Original wave and frequencies
'''
sample_rate, wave = wavfile.read('InputAudio.wav') # read the audio file
print('Sound Rate =', sample_rate)
t = np.arange(len(wave)) / float(sample_rate) # get time values

# Plot Amplitude vs Time
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, wave)
plt.xlabel('Tempo')
plt.ylabel('Amplitude')

# Fourier transform
magnitude, average = fourier_transform(wave.copy())
normalized_magnitude = magnitude / len(wave)

# Build frequencies array
indexes = np.arange(len(normalized_magnitude)) # indexes for all the samples
time_array = len(normalized_magnitude) / float(sample_rate)
frq_array = indexes / float(time_array)  # two sides frequency range
adjust_frq_array = frq_array[range(len(normalized_magnitude) // 2)]  # one side frequency range
adjust_normalized_magnitude = abs(normalized_magnitude[range(len(normalized_magnitude) // 2)])

# Plot Magnitude vs Frequencies
plt.subplot(2, 1, 2)
plt.plot(adjust_frq_array, adjust_normalized_magnitude, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.xticks([x for x in range(0,21000,2000)])
plt.show()

# search for key frequencies
frequencies = [(mag, frq) for mag,frq in zip(normalized_magnitude,frq_array)] # magnitudes with frequencies tuples
key_frqs = sorted(frequencies, reverse=True, key=lambda x: x[0])
# print(key_frqs[:10])
# Key frequencies: 262 , 524 , 785 , 1049

'''
Filtered frequencies and result audio back from inverse fourier
'''
# frequencies filter
min_frqs = [260, 522, 783, 1047]
max_frqs = [264, 526, 787, 1051]
filtered_magnitude = magnitude.copy()
for i in range(len(frq_array)):
    inside_band = False
    for min, max in zip(min_frqs,max_frqs): # check if it is whithin one of the bands
        if frq_array[i] >= min and frq_array[i] <= max:
            inside_band = True
            break
    if not inside_band:
        filtered_magnitude[i] = 0
normalized_magnitude = filtered_magnitude / len(wave) # normalize magnitude to plot

# Inverse Fourier Transform
filtered_wave = scipy.fft.ifft(filtered_magnitude)
filtered_wave = filtered_wave + average # shift back

# Plot Amplitude vs Time
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, filtered_wave)
plt.xlabel('Tempo')
plt.ylabel('Amplitude')

# Build frequencies array
indexes = np.arange(len(normalized_magnitude)) # indexes for all the samples
time_array = len(normalized_magnitude) / float(sample_rate)
frq_array = indexes / float(time_array)  # two sides frequency range
adjust_frq_array = frq_array[range(len(normalized_magnitude) // 2)]  # one side frequency range
adjust_normalized_magnitude = abs(normalized_magnitude[range(len(normalized_magnitude) // 2)])

# Plot Magnitude vs Frequencies
plt.subplot(2, 1, 2)
plt.plot(adjust_frq_array, adjust_normalized_magnitude, 'b')
plt.xlabel('Freq (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.xticks([x for x in range(0,21000,2000)])
plt.show()

# save new filtered wave as a .wav audio file
audio = filtered_wave.astype('int16')
scipy.io.wavfile.write('notaDoBandPass.wav', sample_rate, audio)


'''
Waves format comparison
'''
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, wave)
plt.xlabel('Tempo')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, filtered_wave)
plt.xlabel('Tempo')
plt.ylabel('Amplitude')

plt.show()