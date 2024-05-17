import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft2, ifft2
from tqdm import tqdm
from random import random

Nx = 128
Ny = 128
Lx = 2 * np.pi
Ly = 2 * np.pi
dx = Lx / Nx
dy = Ly / Ny

dt = 0.01
t = 100

c = np.ones((Nx, Ny, int(t/dt))) * 0.7
h = np.ones((Nx, Ny, int(t/dt))) * 0.2
w = np.ones((Nx, Ny, int(t/dt))) * 0.1

theta = np.linspace(0, Lx/Nx, Nx)
'''
for i in range(Nx):
    for j in range(Ny):
        c[i, j, 0] = np.sin(2 * dx * i) * np.sin(dy * j) * 2
        h[i, j, 0] = np.sin(dx * i) * np.sin(dy * j) * 2
        w[i, j, 0] = np.sin(dx * i) * np.sin(dy * j) * 1
'''
'''
for i in range(int(Nx/10), int(Nx/5)):
    for j in range(int(Nx/10), int(Nx/5)):
        c[i, j, 0] = c[i, j, 0] + np.sin(5 * dx * i) * np.sin(5 * dy * j) * 0.2
'''    

'''
for i in range(Nx):
    for j in range(Ny):
        c[i, j, 0] = c[i, j, 0] + np.sin(20 * dx * i) * np.sin(20 * dy * j) * 0.01
        w[i, j, 0] = w[i, j, 0] + np.cos(20 * dx * i) * np.cos(20 * dy * j) * 0.01
        h[i, j, 0] = h[i, j, 0] + np.sin(20 * dx * i) * np.cos(20 * dy * j) * 0.01
'''

for i in range(Nx):
    for j in range(Ny):
        c[i, j, 0] = c[i, j, 0] + random() * 0.2
        w[i, j, 0] = w[i, j, 0] + random() * 0.2
        h[i, j, 0] = h[i, j, 0] + random() * 0.2

Dc = 0.05
Dh = 0.005
Dw = 0.005

A = 0.3
B = 6

ck = fft2(c[:, :, 0])
hk = fft2(h[:, :, 0])
wk = fft2(w[:, :, 0])

kx = np.fft.fftfreq(Nx, dx) * np.pi * 2
ky = np.fft.fftfreq(Ny, dy) * np.pi * 2
k2 = np.zeros((Nx, Ny), float)

for i in range(Nx):
    for j in range(Ny):
        k2[i, j] = kx[i]**2 + ky[j]**2



# Initialize

# Linear term for c, h, w
Lc = Dc * (-k2) * dt
Lh = Dh * (-k2) * dt
Lw = Dw * (-k2) * dt

# Nonlinear term, N(t), N(t-\delta t) for c, h, w
N_c = fft2(A * w[:, :, 0] - B * c[:, :, 0] * h[:, :, 0]**2)
N_h = fft2(B * c[:, :, 0] * h[:, :, 0]**2 - h[:, :, 0] * w[:, :, 0])
N_w = fft2(h[:, :, 0] * w[:, :, 0] - A * w[:, :, 0] - B * c[:, :, 0])

ck = (Lc * ck + N_c) * dt + ck
hk = (Lh * hk + N_h) * dt + hk
wk = (Lw * wk + N_w) * dt + wk

c[:,:,0] = np.real(ifft2(ck))
h[:,:,0] = np.real(ifft2(hk))
w[:,:,0] = np.real(ifft2(wk))


# Iterate from t = dt
for i in tqdm(range(1, int(t / dt)), position=0):
    c[:,:,i] = np.real(ifft2(ck))
    h[:,:,i] = np.real(ifft2(hk))
    w[:,:,i] = np.real(ifft2(wk))
    
    # Nonlinear term, N(t), N(t-\delta t) for c, h, w
    N_c_past = N_c
    N_c = fft2(A * w[:, :, i] - B * c[:, :, i] * h[:, :, i]**2)
    N_h_past = N_h
    N_h = fft2(B * c[:, :, i] * h[:, :, i]**2 - h[:, :, i] * w[:, :, i])
    N_w_past = N_w
    N_w = fft2(h[:, :, i] * w[:, :, i] - A * w[:, :, i] - B * c[:, :, i])
    
    ck = ((1 + Lc * (1/2)) * ck + (3 * N_c - 3 * N_c_past) * (1/2) * dt) / (1 - Lc * (1/2))
    hk = ((1 + Lh * (1/2)) * hk + (3 * N_h - 3 * N_h_past) * (1/2) * dt) / (1 - Lh * (1/2))
    wk = ((1 + Lw * (1/2)) * wk + (3 * N_w - 3 * N_w_past) * (1/2) * dt) / (1 - Lw * (1/2))

if np.isnan(ck).any():
    print("Error, encounter NaN")


for i in range(t):
    plt.pcolormesh(c[:, :, i])
    #plt.imshow(c[:, :, i])
    colorbar = plt.colorbar()

    str = f"Time = {(i):.2f}"
    plt.title(str)
    plt.show(block=False)
    print(sum(sum(c))/Nx/Ny)
    plt.pause(0.1)
    colorbar.remove()

plt.pcolormesh(c[:, :, -1])
colorbar = plt.colorbar()

plt.show()