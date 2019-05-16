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

def assemble_M(Nx):
    print ("assemblage de Mh la matrice de masse, pour Nx = {}".format(Nx))
    row = list()
    col = list()
    data = list()

    h = 1./Nx

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


def assemble_A_DF(Nt,Nx):

    """
    A_{i,i} = 1 + 2*Dt /(h*h)
    A_{i,i-1} = - Dt /(h*h)  (si i > 1)
    A_{i,i+1} = - Dt /(h*h)  (si i < Nx-1)
    """

    print ("assemblage de la matrice A_DF, pour Nt = {0} et Nx = {1}".format(Nt,Nx))
    row = list()
    col = list()
    data = list()

    h = 1./Nx
    Dt = 1./Nt
    a = Dt/(h*h)

    for i in range(1,Nx):
        # range(1, Nx) = { 1, ..., Nx-1 }
        
        if i > 1:
            row.append(i-1)  
            col.append(i-2)   
            data.append(-a)  # A_{i,i-1}  (j=i-1)
    
        row.append(i-1)
        col.append(i-1)
        data.append(1+2*a)       # A_{i,i}

        if i < Nx-1:
            row.append(i-1)  
            col.append(i)  
            data.append(-a)   # A_{i,i+1}    (j=i+1)    

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
    Ah = Mh + nu*Dt*Kh  # matrice a inverser dans le schema en temps

    print("calcul des coefficients de la solution initiale u^0")  
    for i in range(1,Nx):
        u_tx[0,i-1] = u_ini(i*h)
    
    if nb_images >= 1:
        plot_sol(u_tx,n=0, name="sol")

    fh = assemble_fh(Nx)

    print("schema numerique: calcul des solutions u^n pour n = 1, ..., Nt ")  
    for n in range(1,Nt+1):            
        
        b_n = Dt*fh + Mh.dot(u_tx[n-1,:])
        u_tx[n,:] = sparse.linalg.spsolve(Ah, b_n)  # calcule u tel que Ah * u = b_n
        
        # visualisation 
        if nb_images >= 1:
            if n%periode_images == 0 or n == Nt:
                plot_sol(u_tx, n=n, name="sol")    

    print ("Ok: la solution complete est calculee pour ces parametres.")
    return u_tx


def calcul_schema_DF(Nt=None, Nx=None, nb_images=0):
    """
    Fonction qui applique un schema DF implicite, pour un choix des parametres Nt et Nx
    
    Dans cette fonction la solution approchee est calculee 
    et ses valeurs sont stockes dans un tableau temps-espace, sous la forme 
        
        u^n_j = u_tx[n,j-1]     pour n = 0, ..., Nt et j = 1, ..., Nx-1
    
    schema DF:

    (u^{n}_j - u^{n-1}_j)/Dt - (u^{n}_{j+1} - 2*u^{n}_j + u^{n}_{j-1})/(h*h) = f(i*h)

    ie
    
    u^{n}_j - (u^{n}_{j+1} - 2*u^{n}_j + u^{n}_{j-1})*Dt /(h*h) = u^{n-1}_j + Dt*f(i*h)

    donc on doit resoudre

    A u^n = b

    avec (pour i = 1, .., Nx-1 )

    A_{i,i} = 1 + 2*Dt /(h*h)
    A_{i,i-1} = - Dt /(h*h)  (si i > 1)
    A_{i,i+1} = - Dt /(h*h)  (si i < Nx-1)

    et 

    b_i = u^{n-1}_i + Dt*f(i*h)

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
    b_n = numpy.zeros(Nx-1)

    if nb_images >= 1:
        periode_images = int(Nt*1./nb_images)
    else:
        periode_images = 2*T
        

    print("calcul de la matrice du schema")  
    Ah = assemble_A_DF(Nt,Nx)
    

    print("calcul des coefficients de la solution initiale u^0")  
    for i in range(1,Nx):
        u_tx[0,i-1] = u_ini(i*h)
    
    if nb_images >= 1:
        plot_sol(u_tx,n=0, name="sol")

    print("schema numerique: calcul des solutions u^n pour n = 1, ..., Nt ")  
    for n in range(1,Nt+1):            
        
        for i in range(1,Nx-1):
            b_n[i-1] = u_tx[n-1,i-1] + Dt*f(i*h)
        u_tx[n,:] = sparse.linalg.spsolve(Ah, b_n)  # calcule u tel que Ah * u = b_n
        
        # visualisation 
        if nb_images >= 1:
            if n%periode_images == 0 or n == Nt:
                plot_sol(u_tx, n=n, name="sol")    

    print ("Ok: la solution complete est calculee pour ces parametres.")
    return u_tx


def calcul_erreurs(u_tx, erreur_df_explicite=True):
    """
    dans cette fonction on calcule des erreurs de troncature locales, a partir d'une solution approchee
    """
    print("calcul des erreurs pour une solution approchee...")

    Nt, Nx = get_dimensions(u_tx)
    h = 1./Nx
    Dt = T*1./Nt

    err_tx = numpy.zeros((Nt+1,Nx-1))
    err_norm = numpy.zeros((Nt+1))

    # on calcule
    # err_{n,i} = ( u_{n+1,i} - u_{n,i} )/Dt - ( u_{n,i+1} - 2*u_{n,i} + u_{n,i-1} )/(h*h) 
    # pour  n = 0, .., Nt-1  et  i = 2, ..., Nx-2
    
    for n in range(0,Nt):
        for i in range(2,Nx-1):                    

            if erreur_df_explicite:
                # ici on utilise (comme le demandait le sujet) un schema explicite pour definir l'erreur locale.
                err_tx[n,i-1] = (
                    ( u_tx[n+1,i-1]-u_tx[n,i-1] ) / Dt 
                    - ( u_tx[n,i] - 2*u_tx[n,i-1] + u_tx[n,i-2]) / (h*h)
                    - f(i*h)
                    )

            else:            
                # pour comparer, on utilise ici le schema implicite centre pour definir l'erreur locale.
                # On peut alors verifier sur les courbes d'erreur que, sans surprise, l'erreur calculee par 
                # cet indicateur est nulle (a la precision machine pres) lorsque la solution approchee u_tx 
                # est calculee avec le meme schema implicite centre... 
                # Et qu'en est-il pour la solution calculee avec un schema EF ?
                err_tx[n+1,i-1] = (
                    ( u_tx[n+1,i-1]-u_tx[n,i-1] ) / Dt 
                    - ( u_tx[n+1,i] - 2*u_tx[n+1,i-1] + u_tx[n+1,i-2]) / (h*h)
                    - f(i*h)
                    )

        # plot_sol(err_tx, n=n, name="err")    
        # calcul de la norme L2 des erreurs au temps n
        tmp_sum = 0
        for i in range(1,Nx):
            tmp_sum += (err_tx[n,i-1])**2
        err_norm[n] = numpy.sqrt(h*tmp_sum)
        
        # print("err_norm[n] = {}".format(err_norm[n]))
    print("Ok: le calcul des erreurs est fait")    
    return err_norm

# u_tx = calcul_schema_EF(Nt=10, Nx=50, nb_images=10)

# calcul_erreurs(u_tx)
# exit()

errors = []
labels = []
for (Nt,Nx) in [(50,50),(100,100),(200,200)] : # ,(300,300)]: #[(10,10), (50,50)]:  #(100,100),(200,200),()]:
    u_tx = calcul_schema_EF(Nt, Nx) #, nb_images=5)
    err_norm = calcul_erreurs(u_tx, erreur_df_explicite=True)
    errors.append(err_norm)
    labels.append(repr(Nt)+"_"+repr(Nx))

plot_t_curves(
    array1=errors[0], label1=labels[0], 
    array2=errors[1], label2=labels[1], 
    array3=errors[2], label3=labels[2], 
    name="local error norms", save_fn="EF_errors", logscale=True)



errors = []
labels = []
for (Nt,Nx) in [(50,50),(100,100),(200,200)] : # ,(300,300)]: #[(10,10), (50,50)]:  #(100,100),(200,200),()]:
    u_tx = calcul_schema_DF(Nt, Nx) #, nb_images=5)
    err_norm = calcul_erreurs(u_tx, erreur_df_explicite=True)
    errors.append(err_norm)
    labels.append(repr(Nt)+"_"+repr(Nx))

plot_t_curves(
    array1=errors[0], label1=labels[0], 
    array2=errors[1], label2=labels[1], 
    array3=errors[2], label3=labels[2], 
    name="local error norms", save_fn="DF_errors", logscale=True)


