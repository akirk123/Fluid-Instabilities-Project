import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy import linalg
from matplotlib import pyplot as plt
import scipy.integrate as integrate


n=150
##What is the maximum radius evaluating to?
rmax=2.

dx=rmax/float(n)
## What is the value of k?
k=1.

##What is the adiabatic index?
gamma=5./3.

##What is the viscosity?
def mu(R):
    return 0.1

##What is the initial velocity profile?
def u_phi(R):
     # return 4*R
    # return 4.*R*(1.01-np.tanh((R-1.)/0.01))
    if R<1.25:    
        return 2*R*(1.01+np.tanh((R-1.)/0.01))
    if R>=1.25:
        return 2*R*(1.01-np.tanh((R-1.5)/0.01))
    
## What is the initial velocity profile differentiated wrt R?
def u_phiprime(R):
    return 4.
    # return 4.*(1.01-np.tanh(100.0*R-100.0)-100.0*R+100.0*R*np.tanh(100.0*R-100.0)**2)
    if R<1.25:    
        return 2*(np.tanh(100*(R-1.))+100*R/(np.cosh(100-100*R)*np.cosh(100-100*R))+1.01)
    if R>=1.25:
        return 2*(np.tanh(100*(R-1.5))+100*R/(np.cosh(150-100*R)*np.cosh(100-100*R))+1.01)
           
##What is the initial density profile?
def densinit(R):
    return 1.
    
##What is the initial density profile differentiated wrt R?
def densinitprime(R):
    return 0

## Need to integrate to find the pressure
def pressure(R):
    result = integrate.quad(lambda x: densinit(x)*(u_phi(x)*(u_phi(x))/x), dx, R)
    return(100.+result[0])
    
def pressureprime(R):
    return densinit(R)*u_phi(R)*u_phi(R)/float(R)
    
## What is the sound speed squared?
def cs2(R):
    return(gamma*pressure(R)/densinit(R))

##What is the sound speed?
def cs(R):
    return (cs2(R))**(0.5)
    
##What is the sound speed differentiated wrt R?
def csprime(R):
    if cs(R)==0:
        return 0.
    return (gamma)*(densinit(R)*pressureprime(R)-densinitprime(R)*pressure(R))/(2.*densinit(R)*densinit(R)*cs(R))

##This will be the matrix to fill in the gaps in the matrix with zeros
zero = np.zeros((n,n), dtype=complex)

##Discretising the radius from dx to the maximum radius
p=np.arange(dx,rmax+dx, dx)
l=p.tolist()
def derivative(f,a,method='central',h=0.01):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
 

def mach2(R):
    return (u_phi(R)*u_phi(R)/float(cs2(R)))

def lnphi(R):
    return math.log(u_phi(R))*(u_phi(R))

def der(i):
    return(i*derivative(lnphi,i,h=0.1))
# t = np.linspace(0.5,rmax+dx,dx)
# plt.plot(mach2(t),t,'-',color=(0.55,0,0.55),label='Profile')
# plt.plot(der(t),t,'-',color=(0.55,0,0.55),label='Step Profile')
# plt.title('Initial Angular Velocity Profile')
# plt.xlabel('R')
# plt.ylabel('$\Omega(R)$')
# plt.legend(loc=2)
# plt.show()

final=[]
final1=[]

t = np.arange(0.15, 2., 0.001)
for i in t:
    final.append(der(i))
    
for i in t:
    final1.append(mach2(i))
# red dashes, blue squares and green triangles
plt.plot(t, final, 'b-',label='$d\ln\psi/d\ln r$')
plt.plot(t,final1,'g-', label='$M^2$')
plt.title('Rayleighs Criterion for Double Step Profile')
plt.legend(loc=3)
plt.show()

# for r in l:
#     if (derivative(lnphi,r,h=0.0001)/float(r))<(mach2(r)):
#         print('Instability')
#     # print(derivative(lnphi,r,h=0.0001)/float(r))
#     # print(mach2(r))