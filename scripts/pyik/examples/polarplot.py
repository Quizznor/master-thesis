import pylab
# make color-filled plot of polar coordinate data array.
# define coordinates.
theta = pylab.linspace(0.,2.*pylab.pi,1001)
R = pylab.linspace(0.,pylab.pi,1001)
Y,X = pylab.meshgrid(R, theta)
# data to plot.
Z = pylab.sin(Y)*pylab.sin(4*X) + pylab.exp(-(Y**2/4))
# create subplot.
ax = pylab.subplot(111,polar=True)
# mesh fill.
ax.pcolormesh(X,Y,Z)
# make sure aspect ratio preserved
ax.set_aspect('equal')
pylab.show()

