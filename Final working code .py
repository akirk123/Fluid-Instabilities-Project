# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
from scipy import linalg
from matplotlib import pyplot as plt
import scipy.integrate as integrate


n=400
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
    # return 10*R
    return 30*R*(1.01-np.tanh((R-1.)/0.01))
    # if R<1.25:    
    #     return 30*R*(1.01+np.tanh((R-1.)/0.01))
    # if R>=1.25:
    #     return 30*R*(1.01-np.tanh((R-1.5)/0.01))
    
## What is the initial velocity profile differentiated wrt R?
def u_phiprime(R):
    # return 10.
    return 30*(1.01-np.tanh(100.0*R-100.0)-100.0*R+100.0*R*np.tanh(100.0*R-100.0)**2)
    # if R<1.25:    
    #     return 30*(np.tanh(100*(R-1.))+100*R/(np.cosh(100-100*R)*np.cosh(100-100*R))+1.01)
    # if R>=1.25:
    #     return 30*(np.tanh(100*(R-1.5))+100*R/(np.cosh(150-100*R)*np.cosh(100-100*R))+1.01)
    
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

def Gdiagfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*((k**2)+2./(dx**2.)+1./(i**2.)-1./(dx*i)))
    return ret
    
def Gupfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*((-1./((dx)**2))))
    return ret
    
def Gdownfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*(-1./((dx)**2)+1./(dx*i)))
    return ret

def Hdiagfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*((k**2)+2./(dx*dx) +1./(dx*i)))
    return ret
    
def Hupfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*((-1./(dx*dx))))
    return ret
    
def Hdownfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*(-1./(dx*dx)-1./(i*dx)))
    return ret
    
def Mdiagfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*(-k*k-2./(dx*dx)+1./(i*dx)))
    return ret
    
def Mupfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*(1./(dx*dx)))
    return ret
    
def Mdownfunction(list):
    ret = []
    for i in list:
        ret.append(mu(i)/float(densinit(i))*(1./(dx*dx)-1./(dx*i)))
    return ret

def Afunction(list):
    ret = []
    for i in list:
        ret.append(-2*u_phi(i)/float(i))
    return ret
    
def Bdiagfunction(list):
    ret = []
    for i in list:
        ret.append(2*cs(i)*csprime(i)/float(densinit(i))-u_phi(i)*u_phi(i)/float(densinit(i)*i) +cs(i)*cs(i)/float(densinit(i)*dx))
    return ret
    
def Bupfunction(list):
    ret = []
    for i in list:
        ret.append(0.)
    return ret
    
def Bdownfunction(list):
    ret = []
    for i in list:
        ret.append(-cs(i)*cs(i)/float(densinit(i)*dx))
    return ret
    
def Cfunction(list):
    ret = []
    for i in list:
        ret.append(u_phiprime(i)+u_phi(i)/float(i))
    return ret
    
def Dfunction(list):
    ret = []
    for i in list:
        ret.append(1j*k*cs(i)*cs(i)/densinit(i))
    return ret
    
def Ediagfunction(list):
    ret = []
    for i in list:
        ret.append(densinit(i)/float(i)+densinitprime(i) +densinit(i)/float(dx))
    return ret
    
def Eupfunction(list):
    ret = []
    for i in list:
        ret.append(0.)
    return ret
    
def Edownfunction(list):
    ret = []
    for i in list:
        ret.append(densinit(i)/float(-dx))
    return ret
    
def Ffunction(list):
    ret = []
    for i in list:
        ret.append(1j*k*densinit(i))
    return ret


## Here I define the individual blocks that will build up the large matrix
A = np.zeros((n,n), dtype=complex)
i,j = np.indices(A.shape)
A[i==j] = Afunction(l)
A[0,0]=0
A[1,0]=0

B = np.zeros((n,n), dtype=complex)
i,j = np.indices(B.shape)
B[i==j] = Bdiagfunction(l)
upB=l[:]
downB=l[:]
del upB[-1]
del downB[0]
B[i==j-1]=Bupfunction(upB)
B[i==j+1]=Bdownfunction(downB)

C = np.zeros((n,n),dtype=complex)
i,j = np.indices(C.shape)
C[i==j] = Cfunction(l)
C[-1,-1]=0
C[-2,-1]=0
C[0,0]=0
C[1,0]=0

D = np.zeros((n,n),dtype=complex)
D=D.astype(complex)
i,j = np.indices(D.shape)
D[i==j] = Dfunction(l)

E = np.zeros((n,n),dtype=complex)
i,j = np.indices(E.shape)
E[i==j] = Ediagfunction(l)
upE=l[:]
downE=l[:]
del upE[-1]
del downE[0]
E[i==j-1]=Eupfunction(upE)
E[i==j+1]=Edownfunction(downE)
E[-1,-1]=0
E[-2,-1]=0
E[0,0]=0
E[1,0]=0

F = np.zeros((n,n),dtype=complex)
F=F.astype(complex)
i,j = np.indices(F.shape)
F[i==j] = Ffunction(l)

G = np.zeros((n,n),dtype=complex)
i,j = np.indices(G.shape)
G[i==j] = Gdiagfunction(l)
upG=l[:]
downG=l[:]
del upG[-1]
del downG[0]
G[i==j-1]=Gupfunction(upG)
G[i==j+1]=Gdownfunction(downG)
G[-1,-1]=0
G[-2,-1]=0
G[0,0]=0
G[1,0]=0

H = np.zeros((n,n),dtype=complex)
i,j = np.indices(H.shape)
H[i==j] = Hdiagfunction(l)
upH=l[:]
downH=l[:]
del upH[-1]
del downH[0]
H[i==j-1]=Hupfunction(upH)
H[i==j+1]=Hdownfunction(downH)
H[0,0]=0
H[1,0]=0

M = np.zeros((n,n),dtype=complex)
i,j = np.indices(M.shape)
M[i==j] = Mdiagfunction(l)
upM=l[:]
downM=l[:]
del upM[-1]
del downM[0]
M[i==j-1]=Mupfunction(upM)
M[i==j+1]=Mdownfunction(downM)
#print(M)

N=np.concatenate((G,A,zero,B), axis=1)
Z=np.concatenate((C,H,zero,zero), axis=1)
L=np.concatenate((zero,zero,M,D), axis=1)
P=np.concatenate((E,zero,F,zero), axis=1)
final=np.concatenate((N,Z,L,P), axis=0)


eigenValues, eigenVectors = linalg.eig(final, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)
idx = eigenValues.argsort()[::1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print(eigenValues[2*n-150:2*n-50])

plt.plot(eigenValues.real, eigenValues.imag,'o',markerfacecolor=(0.41,0.16,0.38),fillstyle='full',markerfacecoloralt=(0.55,0,0.55),markersize=4)
plt.title('Eigenvalues for Step Profile')
# plt.legend(loc=1)
axhline(y=0,color=(0.55,0,0.55))
plt.xlabel('Eigenvalue Real Part')
plt.ylabel('Eigenvalue Imaginary Part')
plt.show()
plt.show()

        
##print(eigenVectors[0:10])
#c=np.arange(n+85, n+95)

c=[2*n-12]
# print(c)

m=n
print(eigenValues[c])
for k1 in c:
    print(k1)
    plt.figure()
    xR=[0]
    yR=[0]
    zR=[0]
    vR=[0]
    xIm=[0]
    yIm=[0]
    zIm=[0]
    vIm=[0]
    for i in range(0,m-1):
        xIm.append(eigenVectors[i][k1].imag)
        xR.append(eigenVectors[i][k1].real)
    for i in range(m,2*m-1):
        yIm.append(eigenVectors[i][k1].imag)
        yR.append(eigenVectors[i][k1].real)
    for i in range(2*m,3*m-1):
        zIm.append(eigenVectors[i][k1].imag)
        zR.append(eigenVectors[i][k1].real)
    for i in range(3*m,4*m-1):
        vIm.append(eigenVectors[i][k1].imag)
        vR.append(eigenVectors[i][k1].real)
    xR.append(0)
    yR.append(0)
#    zR.append(zR[-1])
    vR.append(0)
    xIm.append(0)
    yIm.append(0)
#    zIm.append(zIm[-1])
    vIm.append(0)
#    print(shape(x))
    t = np.arange(0,rmax+dx,dx)
    print(eigenValues[k1])
    plt.plot(t,xR,'r-',label='Perturbation in $\hat{r}$')
    plt.plot(t,xIm,'r:')
    plt.plot(t,yR,'g-',label='Perturbation in $\hat{\phi}$')
    plt.plot(t,yIm,'g:')
    plt.plot(t[:-1],zR,'b-',label='Perturbation in $\hat{z}$')
    plt.plot(t[:-1],zIm,'b:')
    plt.plot(t,vR,'k-',label='Perturbation in density')
    plt.plot(t,vIm,'k:',)
    plt.grid(False)
    plt.legend(loc=3)
    plt.title('Eigenmode for Step Profile with Eigenvalue..')
    plt.show()