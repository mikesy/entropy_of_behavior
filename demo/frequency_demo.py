import numpy as np
import matplotlib.pyplot as plt


#TODO super mess, clean 

total_sample_time = 2.0
num_steps = 150
dt = total_sample_time/num_steps
plt.figure(0)
t = np.arange(num_steps)*dt
x = np.sin(t*2.0*np.pi)
x = np.ones(num_steps)
sp = np.fft.fft(x)
freq = np.fft.fftfreq(num_steps)
plt.plot(freq, sp)

plt.figure(1)
x = np.cos(20*2*np.pi*t)
y = np.sin(30*2*np.pi*t)
u = np.transpose(np.array([x, y]))
# u = np.array([x,y])
sp = np.fft.fft(u, axis=0, n=num_steps)
freq = np.fft.fftfreq(num_steps, d=dt)
plt.plot(freq, sp[:, 0])
plt.plot(freq, sp[:, 1])
# plt.legend(['0','1'])
plt.figure(2)

plt.plot(abs(freq), (2/num_steps)*np.linalg.norm(sp, axis=1))
# plt.figure(3)

# plt.plot(freq,x)
norm_sp = np.linalg.norm(sp, axis=1)
threshold = 0.5*np.amax(norm_sp)

mask = np.where(norm_sp > threshold)

peaks = freq[mask]
mask = np.where(peaks > 0)
peaks = peaks[mask]

# print(peaks)
# plt.figure(3)
# plt.hist(peaks)
# plt.show()
