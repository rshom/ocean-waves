'''Ocean Wave Environment using realistic waves which are affected by
depth.

'''

import numpy as np

from scipy import signal
from scipy.fftpack import fft

from numpy import pi

GRAVITY = 9.81
DENSITY = 1000

class WaveEnv:

    def __init__(self, U10=0, spectrum = None):

        if spectrum:
            self.omega = spectrum[0]
            self.phi = -pi
            self.A = spectrum[1]
        else:
            self.omega, self.S = pierson_moskowitz(U10)
            self.phi = 2*pi*np.random.rand(1,len(self.omega)) # randomized phase
            domega = np.diff(self.omega, prepend=0)
            self.A = np.sqrt(2*self.S*domega)

    def depth_loss(self,z):
        l = 2*pi*GRAVITY/self.omega**2
        return np.exp(2*pi*z/l)

    def displacement(self,t,z=0):
        B = np.sin(self.omega*t+self.phi)
        x = -np.sum(self.depth_loss(z)*self.A*B)
        return x

    def velocity(self,t,z=0):
        B = self.omega*np.cos(self.omega*t+self.phi)
        v = -np.sum(self.depth_loss(z)*self.A*B)
        return v

    def acceleration(self,t,z=0):
        B = -self.omega**2*np.sin(self.omega*t+self.phi)
        a = -np.sum(self.depth_loss(z)*self.A*B)
        return a

    def depth_spectrum(self,z=0):
        return self.depth_loss(z)**2*self.S

def pierson_moskowitz(U10):
    '''Returns the spectrum of waves'''

    k0 = np.logspace(-2,0,100)
    w = np.sqrt(k0*GRAVITY)     # omega
    U19p5 = 1.026*U10
    beta = .74
    w0 = GRAVITY/U19p5          # limiting frequency
    alpha = 8.31e-3
    A = alpha*GRAVITY**2
    H = 1                       # significant wave height
    B = beta*w0**4

    # S = A/w**5*np.exp(-.032*(GRAVITY/H/w**2))**2
    # S = A/w**5*np.exp(-B/w**4)
    S = A/w**5*np.exp(-beta*(w0/w)**4)

    Hs = 0.21*U19p5**2/GRAVITY
    wm = 0.4*np.sqrt(GRAVITY/Hs)
    S = A/w**5 *np.exp(-5/4*(wm/w)**4)

    return w,S

# TODO: define more spectrums

if __name__=='__main__':

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    windSpeed = 8
    waves = WaveEnv(U10=windSpeed)
    #waves = WaveEnv(spectrum=(2,1))

    fs = 2*np.max(waves.omega)/(2*pi)
    fs = 10
    dt = 1/fs
    T = 500
    t = np.arange(0,T,dt)

    wave_x = np.vectorize(waves.displacement)
    wave_v = np.vectorize(waves.velocity)
    wave_a = np.vectorize(waves.acceleration)

    fig,axs = plt.subplots(3,1,sharex=True,sharey=False)
    axs[0].set_title('Ocean Wave Action (U10={} m/s)'.format(windSpeed))
    axs[0].set_ylabel('Displacement (m)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[2].set_ylabel('Acceleration (m/s^2)')
    axs[-1].set_xlabel('Time (s)')

    fig,axs2 = plt.subplots(2,1,sharex=True,sharey=True)
    axs2[0].set_title('Predicted Ocean Wave Spectrum')
    axs2[1].set_title('Simulated Ocean Wave Spectrum')
    axs2[-1].set_xlabel('Freq ($Hz$)')
    axs2[0].set_ylabel('Power Density ($m^2/Hz)$)')
    axs2[1].set_ylabel('Power Density ($m^2/Hz)$)')

    for z in [0,-1,-5,-10]:
        axs[0].plot(t,wave_x(t,z),label='z={} m'.format(z))
        axs[1].plot(t,wave_v(t,z),label='z={} m'.format(z))
        axs[2].plot(t,wave_a(t,z),label='z={} m'.format(z))

        Sp = waves.depth_spectrum(z)
        freqs = waves.omega/(2*pi)
        axs2[0].plot(freqs,Sp, label='z={} m'.format(z))

        sig = wave_x(t,z)

        # freqs, Sm = signal.welch(sig, fs)
        # Sm = Sm/(2*pi)
        # axs2[1].plot(freqs,Sm,label='z={} m'.format(z))

        # ???: these scales do not seem to be working correctly. The
        # shapes are correct. I do not think that the peodogram should
        # be normalized either. My understanding is that this shows
        # the periodogram shows the total energy in the signal since
        # it scales with length. Therefore, I scale by fs/T which
        # seems to put the result in a reasonable scale.

        freqs = waves.omega
        pgram = signal.lombscargle(t, sig, freqs, normalize=False)
        axs2[1].plot(freqs/(2*pi),pgram*fs/T, label='z={} m'.format(z))

        # print(z,np.sum(pgram),np.sum(Sp),np.sum(pgram)/np.sum(Sp))
        # print('Ho = {}'.format(4*np.sqrt(np.mean(sig**2))),.21*windSpeed**2/GRAVITY)
        E = DENSITY*GRAVITY*np.std(sig)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs2[0].legend()
    axs2[1].legend()


    plt.show()

