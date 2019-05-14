"""
schemas EF pour l'equation de la chaleur avec conditions aux bords de type Dirichlet homogenes
"""

import numpy
import pylab
import os

import scipy.sparse as sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

# domaine spatial et maillage
X_min = 0
X_max = 1
Nx = 100
h = 1./Nx
X = numpy.zeros(Nx+1)       # grille
for i in range(0,Nx+1):
    X[i] = X_min + i*h*(X_max-X_min)

# domaine temporel
Nt = 100
T = 1
Dt = T * 1./Nt
    
# parametres de visualisation
n_images = 10
periode_images = int(Nt*1./n_images)

# fonction initiale
def u_ini(x):
    return 0   #return numpy.cos(2*numpy.pi*x)

# fonctions de base
def phi(i,x):
    # calcule phi_i(x)    
    return max(1-abs(x/h-i),0)

# coefs de la solution approchee dans la base nodale: u[j], j = 0, ..., N-2
u = numpy.zeros(Nx-1)       
next_u = numpy.zeros(Nx-1)  # une solution auxiliaire

# evaluation d'une solution a partir de ses coefficients v[j], j = 0, ..., N-2
def eval_v(v,x):
    assert len(v) == Nx-1
    val = 0
    for i in range(1, Nx):
        # note: on pourrait restreindre cette boucle a deux valeurs
        val += v[i-1]*phi(i,x)
    return val

# utilitaire de visualisation d'une solution 
fig = plt.figure()
def plot_sol(nt=None):
    fig.clf()
    message = "Plot solution"
    if nt is None:
        fname = "plot_sol.png"
        title = "approx. solution u_h"
    else:
        fname = "plot_sol_nt="+repr(nt)+".png"
        message += " for nt="+repr(nt)
        title = "approx. solution u_h at t="+repr(nt*Dt)
    print (message + ", min/max = ", min(u), "/", max(u))
    #u_full = numpy.concatenate(([0],u,[0]),axis=0)
    u_full = [eval_v(u,x) for x in X]
    u_min = min(u_full)
    u_max = max(u_full)
    Y_min = u_min - 0.1*(u_max-u_min)
    Y_max = u_max + 0.1*(u_max-u_min)
    plt.xlim(X_min, X_max)
    plt.ylim(Y_min, Y_max)
    plt.xlabel('x')
    plt.title(title)
    # plt.legend(loc="upper center")
    plt.plot(X, u_full, '-', color='k')
    fig.savefig(fname)
    
def assemble_K():
    print ("assemblage de Kh la matrice de raideur (Laplacien) -- a corriger..."  )
    row = list()
    col = list()
    data = list()

    ## exercice: corriger la boucle ci-dessous pour assembler la matrice du laplacien (raideur)

    for i in range(1,Nx):
        # range(1, Nx) = { 1, ..., Nx-1 }

        if i > 1:
            j=i
            row.append(j-2)  
            col.append(j-1)  
            data.append(-Nx)   # A_{j-1,j}    (j=i+1)
    
        row.append(i-1)
        col.append(i-1)
        data.append(2*Nx)       # A_{i,i}
    
        if i > 1:
            row.append(i-1)  
            col.append(i-2)   
            data.append(-Nx)  # A_{i,i-1}  (j=i-1)

    row = numpy.array(row)
    col = numpy.array(col)
    data = numpy.array(data)      
    return (sparse.coo_matrix((data, (row, col)), shape=(Nx-1, Nx-1))).tocsr()


def assemble_M():
    print ("assemblage de Mh la matrice de masse -- a corriger..."  )
    row = list()
    col = list()
    data = list()

    ## exercice: corriger la boucle ci-dessous pour assembler la matrice de masse
    h = 1/Nx

    for i in range(1,Nx):
        # range(1, Nx) = { 1, ..., Nx-1 }
        
        if i > 1:
            j=i
            row.append(j-2)  
            col.append(j-1)  
            data.append(h/6)   # M_{j-1,j}    (j=i+1)
    
        row.append(i-1)
        col.append(i-1)
        data.append(2*h/3)       # M_{i,i}
    
        if i > 1:
            row.append(i-1)  
            col.append(i-2)   
            data.append(h/6)  # M_{i,i-1}  (j=i-1)

    row = numpy.array(row)
    col = numpy.array(col)
    data = numpy.array(data)      
    return (sparse.coo_matrix((data, (row, col)), shape=(Nx-1, Nx-1))).tocsr()

print ("assemblage de la source discrete..."  )
fh = numpy.zeros(Nx-1)
# ici on va calculer fh = (<f,phi_i>)_{i = 1, ..., Nx-1}
for i in range(1,Nx):
    fh[i-1] = 0.5*( phi(i,1./3) + phi(i,2./3) )   


print("calcul de la matrice du schema -- a corriger...")  
Kh = assemble_K()
Mh = assemble_M()
nu = 0.1   # nu: coefficient de diffusion -- on peut tester differentes valeurs de nu et comparer les evolutions de u
Ah = nu*Dt*Kh + Mh  # matrice a inverser dans le schema en temps


# initialisation de la solution
for i in range(1,Nx):
    u[i-1] = u_ini(X[i])
    
plot_sol(0)

## schema numerique
for nt in range(0,Nt):           
    b = Dt*fh + Mh.dot(u)   # b^n = M u^{n-1} + Dt*fh
    next_u[:] = sparse.linalg.spsolve(Ah, b)  # calcule u^n tel que Ah * u^n = b^n (voir ci-dessus)
    u[:] = next_u[:]

    # visualisation 
    if (nt+1)%periode_images == 0 or (nt+1) == Nt:
        plot_sol(nt+1)    

print ("Fini.")
