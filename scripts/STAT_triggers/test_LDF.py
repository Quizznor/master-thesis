import matplotlib.pyplot as plt
import numpy as np

# see Davids Mail from 27/06

def GetLDFSignalWCD(r, sropt, theta):
    ropt=1000.
    rbend=700.
    beta = GetBetaWCD(sropt, theta)
    gamma = GetGammaWCD(sropt, theta)
    return sropt * (r/ropt)**beta * ((r+rbend)/(ropt+rbend))**(beta+gamma)

def GetBetaWCD(S, theta):
    ropt=1000
    a = -3.72
    b = 0.0967
    c = 1.74
    d = -0.242
    e = -0.274
    f = 0.0349
    sectheta = 1/np.cos(theta)
    s = np.log10(S)
    return (a+b*s)+(c+d*s)*sectheta+(e+f*s)*sectheta**2.

def GetGammaWCD(S, theta):
    ropt=1000
    b = -1.87
    c = -0.183
    d = 0.490
    e = -0.0650
    f = -2.272
    g = 4.64
    h = 18.01
    i = -1.95
    costheta = np.cos(theta)
    s = np.log10(S)
    a = np.exp((19.6-2.10*s)*(costheta**2.-(0.483+0.005*s)))
    return b+c*s+(d+e*s)*1/(a+1)+f*costheta**g*1/(np.exp(h*(s-i))+1.) - GetBetaWCD(S, theta)

THETA = np.linspace(10,80,90)
for r in np.linspace(10,3000, 10):
    plt.plot(THETA, GetLDFSignalWCD(r, 1000000, THETA))

plt.show()