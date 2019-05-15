"""
schemas EF et DF pour l'equation de la chaleur
"""

import numpy
import pylab
import os

import scipy.sparse as sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

# temps final 
T = 1

# fonction initiale
def u_ini(x):
    return 0   # return numpy.cos(2*numpy.pi*x)

# source de chaleur
def f(x):
    val = 0
    if 1./3 <= x <= 2./3:
        val = 1
    return val

# fonctions de base
def phi(i,x,Nx):
    # calcule phi_i(x) pour une valeur de Nx donnee   
    y = x*Nx
    if i-1 <= y <= i:
        val = y - (i-1)
    elif i <= y <= i+1:
        val = (i+1) - y 
    else:
        val = 0
    return val # max(1-abs(x*Nx-i),0)

def get_dimensions(v_tx):
    Nt = v_tx.shape[0]-1
    Nx = v_tx.shape[1]+1
    return Nt, Nx

def eval_v(v_tx,n,x):
    """
    Evaluation d'une solution en un point (t_n,x) a partir de ses coefficients sur la grille t-x.
    Les coefficients doivent etre sous la forme 
        v^n_j = v_tx[n,j-1] 
    pour n = 0, ..., Nt et j = 1, ..., Nx-1
    """
    Nt, Nx = get_dimensions(v_tx)
    val = 0
    for i in range(1, Nx):
        # note: on pourrait restreindre cette boucle a deux valeurs
        val += v_tx[n,i-1]*phi(i,x,Nx)
    return val

# utilitaire de visualisation d'une solution 
fig = plt.figure()
def plot_sol(u_tx, n=None, name="sol"):
    fig.clf()
    Nt, Nx = get_dimensions(u_tx)
    h = 1./Nx 
    Dt = T*1./Nt
    if n is None:
        raise ValueError('je ne peux pas tracer la solution u^n si je ne connais pas n')
    else:
        assert 0 <= n <= Nt 
        fname = name+"_n="+repr(n)+".png"
        message = "on trace la solution en n = "+repr(n)
        title = "solution approchee en t="+repr(n*Dt)
    #u_full = numpy.concatenate(([0],u,[0]),axis=0)
    x_grid = [i*h for i in range(0,Nx+1)]
    u_full = [eval_v(u_tx,n,i*h) for i in range(0,Nx+1)]
    u_min = min(u_full)
    u_max = max(u_full)
    Y_min = u_min - 0.1*(u_max-u_min)
    Y_max = u_max + 0.1*(u_max-u_min)
    print(message + ", min/max = ", u_min, "/", u_max)
    plt.xlim(0, 1)
    plt.ylim(Y_min, Y_max)
    plt.xlabel('x')
    plt.title(title)
    # plt.legend(loc="upper center")
    plt.plot(x_grid, u_full, '-', color='k')
    fig.savefig(fname)

def get_times_array_for(array):
    Nta = len(array)-1
    dta = T/Nta
    times = numpy.zeros(Nta+1)
    for n in range(0,Nta+1):
        times[n] = n*dta
    return times

def plot_t_curves(
    array1=None, label1=None, 
    array2=None, label2=None, 
    array3=None, label3=None,
    name="some values along time", save_fn=None, logscale=False
):
    plt.xlabel('t')
    plt.ylabel(name)
    if logscale:
        plt.yscale("log")
    assert array1 is not None
    times = get_times_array_for(array1)
    plt.plot(times, array1, '-', label=label1)
    if array2 is not None:
        times = get_times_array_for(array2)
        plt.plot(times, array2, '-', label=label2)
    if array3 is not None:
        times = get_times_array_for(array3)        
        plt.plot(times, array3, '-', label=label3)
    if save_fn is not None:
        plt.savefig(save_fn)
    if label1 is not None:
        plt.legend()
    plt.show()
    

def assemble_K(Nx):
    print ("assemblage de Kh la matrice du Laplacien, pour Nx = {}".format(Nx))
    row = list()
    col = list()
    data = list()

    for i in range(1,Nx):
        # range(1, Nx) = { 1, ..., Nx-1 }

        for j in range(1,Nx):

            ####    A CORRIGER 
            if i==j:
                row.append(i-1)  
                col.append(j-1)  
                data.append(1)      # K_{i,j}

    row = numpy.array(row)
    col = numpy.array(col)
    data = numpy.array(data)      
    return (sparse.coo_matrix((data, (row, col)), shape=(Nx-1, Nx-1))).tocsr()

def assemble_M(Nx):
    print ("assemblage de Mh la matrice de masse, pour Nx = {}".format(Nx))
    row = list()
    col = list()
    data = list()

    h = 1./Nx

    for i in range(1,Nx):
        # range(1, Nx) = { 1, ..., Nx-1 }
        
        for j in range(1,Nx):

            ####    A CORRIGER 
            if i==j:
                row.append(i-1)  
                col.append(j-1)  
                data.append(1)      # M_{i,j}


    row = numpy.array(row)
    col = numpy.array(col)
    data = numpy.array(data)      
    return (sparse.coo_matrix((data, (row, col)), shape=(Nx-1, Nx-1))).tocsr()

def assemble_fh(Nx):
    """
    calcul des coefs (approches) fh[i-1] = <f,phi_i>   pour i = 1, ..., Nx-1
    """
    print ("assemblage de la source discrete, pour Nx = {}".format(Nx))
    fh = numpy.zeros(Nx-1)
    h = 1./Nx
    for i in range(1,Nx):
        fh[i-1] = f(i*h)*h
    return fh

def calcul_schema_EF(Nt=None, Nx=None, nb_images=0):
    """
    Fonction qui applique le schema EF, pour un choix des parametres Nt et Nx
    
    Dans cette fonction la solution approchee est calculee 
    et ses valeurs sont stockes dans un tableau temps-espace, sous la forme 
        
        u^n_j = u_tx[n,j-1]     pour n = 0, ..., Nt et j = 1, ..., Nx-1
    
    Si on donne un nombre d'images (par defaut nb_images = 0), la solution sera tracee en differents instants
    """

    if Nt is None:
        raise ValueError('le parametre Nt doit etre fourni')
    if Nx is None:
        raise ValueError('le parametre Nx doit etre fourni')

    print ("calcul de la solution u_tx pour les parametres Nt = {0}, Nx = {1} ...".format(Nt,Nx))

    # grille
    h = 1./Nx
    Dt = T*1./Nt
    
    # coefs de la solution approchee dans un tableau t-x (voir plus haut)
    u_tx = numpy.zeros((Nt+1, Nx-1))

    if nb_images >= 1:
        periode_images = int(Nt*1./nb_images)
    else:
        periode_images = 2*T
        
    print("calcul de la matrice du schema")  
    Kh = assemble_K(Nx)
    Mh = assemble_M(Nx)
    nu = 1   # nu: coefficient de diffusion
    Ah = Mh  # matrice a inverser dans le schema en temps           #### A CORRIGER

    print("calcul des coefficients de la solution initiale u^0")  
    for i in range(1,Nx):
        u_tx[0,i-1] = u_ini(i*h)
    
    if nb_images >= 1:
        plot_sol(u_tx,n=0, name="sol")

    fh = assemble_fh(Nx)

    print("schema numerique: calcul des solutions u^n pour n = 1, ..., Nt ")  
    for n in range(1,Nt+1):            
        
        b_n = Mh.dot(u_tx[n-1,:])      ####    A CORRIGER 
        u_tx[n,:] = sparse.linalg.spsolve(Ah, b_n)          # calcule u tel que Ah * u = b_n
        
        # visualisation 
        if nb_images >= 1:
            if n%periode_images == 0 or n == Nt:
                plot_sol(u_tx, n=n, name="sol")    

    print ("Ok: la solution complete est calculee pour ces parametres.")
    return u_tx


def calcul_erreurs(u_tx):
    """
    dans cette fonction on calcule des erreurs de troncature locales, a partir d'une solution approchee
    """
    print("calcul des erreurs pour une solution approchee...")

    Nt, Nx = get_dimensions(u_tx)
    h = 1./Nx
    Dt = T*1./Nt

    err_tx = numpy.zeros((Nt+1,Nx-1))
    err_norm = numpy.zeros((Nt+1))

    # on calcule une erreur de troncature locale
    # err_{n,i} = ...
    # pour  n = 0, .., Nt-1  et  i = 2, ..., Nx-2
    
    for n in range(0,Nt):
        for i in range(2,Nx-1):                    
            err_tx[n,i-1] = 1     ####   A CORRIGER
        
        # plot_sol(err_tx, n=n, name="err")    
        
        # calcul de la norme L2 des erreurs au temps n
        tmp_sum = 0
        for i in range(1,Nx):
            tmp_sum += (err_tx[n,i-1])**2
        err_norm[n] = numpy.sqrt(h*tmp_sum)
        
        # print("err_norm[n] = {}".format(err_norm[n]))
    print("Ok: le calcul des erreurs est fait")    
    return err_norm

    
errors = []
labels = []
for (Nt,Nx) in [(50,50),(100,100),(200,200)]:
    u_tx = calcul_schema_EF(Nt, Nx, nb_images=0)
    err_norm = calcul_erreurs(u_tx)
    errors.append(err_norm)
    labels.append(repr(Nt)+"_"+repr(Nx))

plot_t_curves(
    array1=errors[0], label1=labels[0], 
    array2=errors[1], label2=labels[1], 
    array3=errors[2], label3=labels[2], 
    name="local error norms", logscale=True)


