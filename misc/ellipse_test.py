
import numpy as np
import matplotlib.pyplot as plt

x_data = np.arange(0,100,0.1)

X = np.sin(x_data)
Y = np.sin(x_data+0.1)

isnan = np.isnan(X)
Y = Y[~isnan]
X = X[~isnan]

X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)

A = np.ma.hstack([X**2, X*Y, Y**2, X, Y])
b = np.ones_like(X)
x = np.linalg.lstsq(A, b)[0].squeeze()

# Print the equation of the ellipse in standard form
print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

plt.scatter(X, Y, s=8)

x_coord = np.linspace(-6,6,1000)
y_coord = np.linspace(-6,6,1000)
X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

plt.grid()
plt.plot([-60,60],[-60,60], linestyle='--', color='k')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

a = x[0]
b = x[1]
c = x[2]
d = x[3]
e = x[4]
f = -1

q0 = f*(4*a*c - b**2) - a*np.e**2 + b*d*np.e - c*d**2
q1 = (4*a*c - b**2)**2
q = q0/q1

s = .25*np.sqrt(2*np.abs(q)*np.sqrt(b**2 + (a-c)**2))

smaj = np.sqrt( 2*np.abs(q) * np.sqrt(b**2+(a-c)**2) - 2*q*(a+c) )/8
smin = np.sqrt(smaj**2 - s**2)