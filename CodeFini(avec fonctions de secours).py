import numpy as np
import matplotlib.pyplot as plt





"----------------------------------------------------------------------------------------------------------------------------------------"
"---------------------------------------- Algorithmes de la dichotomie, de Newton dimension 1 et de la secante ----------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------------------------"





def dichotomie(f,a,b,precision=1e-6,n=200):           # n est le nombre d'itérations max (à ne pas dépasser)
    fa = f(a)                                         # On associe à fa et fb les valeurs f(a) et f(b)
    fb = f(b)
    compteur=0                                        # On initialise le compteur 
    assert fa*fb <= 0                                 # Les 2 "assert" permettent de vérifier que les conditions initiales soient bien respectées
    assert a<b
    if fa==0:                                         # Si f(a) ou f(b) sont nulles, on a trouvé notre racine
        return a
    if fb==0:
        return b
    while b-a > 2*precision and compteur<n:           # Tant que la longueur de l'intervalle est trop grande par rapport à la précision (ou que le compteur n'a pas atteint la limite), faire
        m = (a+b)/2                                   # On divise par 2 l'intervalle
        fm = f(m)    
        compteur += 1                                 # A chaque fois qu'on divise par deux l'intervalle, on augmente de 1 le compteur
        if fa*fm<=0:                                  # Si la condition est remplie, alors la partie de gauche comporte une racine car f(a) et f(m) sont de signes opposés
            b, fb = m, fm                             # On s'intéresse alors à la partie de gauche, avec m qui fait office de b (on le met à jours)
        else:
            a, fa = m, fm                             # Sinon, on s'intéresse à la partie de droite
    print(compteur)                                   # On retourne le nombre d'itérations
    return (a+b)/2                                    #On retourne la "racine" trouvé qui respecte la précision voulue





def NewtonR1(f,g,x0,precision=1e-6,n=200):            # n est le nombre d'itérations max (à ne pas dépasser)
    x = x0                                            # x0 est notre point initial
    compteur=0 # 
    while (compteur<n and np.abs(f(x)) > precision):  # Tant qu'on n'a pas |f(x)| < precision et que le nombre d'itérations max n'est pas atteint, faire
        compteur += 1                                 # On augmente le compteur de 1 à chaque itération
        assert g(x) != 0                              # On vérifie que la dérivé ne s'annule pas en x (sinon on ne peut pas diviser)
        x = x - (f(x)/g(x))
    if np.abs(f(x)) > precision:                      # Cas où le nombre d'itérations max a été atteint sans résultat concret
        print("L'algorithme ne converge pas en " + str(n) +" étapes")
        return False
    print(compteur) 
    return x




def secante(f, x0, x1, precision=1e-6, n=200):
    x_prec, x_act = x0, x1
    compteur = 0 
    while compteur < n and np.abs(f(x_act)) > precision:                    # On utilise les même conditions que précédemment
        f_prec = f(x_prec)
        f_act = f(x_act)
        assert((f_act - f_prec)!=0)                                         # On vérifie bien que f(x0) n'est pas égale à f(x1) (pour utiliser la méthode de la sécante)
        x_nouv = x_act - (x_act - x_prec) * (f_act / (f_act - f_prec))
        x_act, x_prec = x_nouv, x_act                                       # On actualise
        compteur += 1                                                       # On augmente de 1 le compteur d'itérations
    if np.abs(f(x_act)) > precision:                                        # Cas où le nombre d'itérations max est atteint
        print("L'algorithme ne converge pas en " + str(n) +" étapes")
        return False
    print(compteur)
    return x_act





"---------------------------------------- Tests des algorithmes de la dichotomie, de Newton dimension 1 et de la secante ---------------------------------------------------------------------"


#Test Newton dans R, Dichotomie et Secante

def f1(x):
    return x**3 - 2*x**2 + x - 1

def g1(x):
    return 3*x**2 - 4*x +1

def f2(x):
    return np.exp(x) - 2

def g2(x):
    return np.exp(x)

def f3(x):
    return np.sin(x) - 0.5

def g3(x):
    return np.cos(x)

def f4(x):
    return np.log(x**2) - 1

def g4(x):
    return 2*x/(x**2)

def f5(x):
    return x**2 - 9

def g5(x):
    return 2*x


#print(NewtonR1(f1,g1,3000))
#print(NewtonR1(f2,g2,700))
#print(NewtonR1(f3,g3,500))
#print(NewtonR1(f4,g4,0.001))
#print(NewtonR1(f5,g5,2000))

#print(dichotomie(f1,-100,100))
#print(dichotomie(f2,-100,100))
#print(dichotomie(f3,-30,20))
#print(dichotomie(f4,-2,50))
#print(dichotomie(f5,0,1000))

#print(secante(f1,-400,100))
#print(secante(f2,-10,10))
#print(secante(f3,-1,1))
#print(secante(f4,0.2,200))
#print(secante(f5,-124,100))




"----------------------------------------------------------------------------------------------------------------------------------------"
"---------------------------------------- Algorithme de Newton-Raphson en dimension n ----------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------------------------"




def approx_jacob(F, X, h=1e-6):
    n = len(X)                                            # On enregiste la dimension du problème (à savoir la taille de X)
    J = np.zeros((n, n))                                  # On initialise la jacobienne approchée (on remplit la tableau de 0)
    for i in range(n):
        dX = np.zeros(n)                
        dX[i] = h

        F_plus = F(X + dX)
        F_moins = F(X - dX)

        J[:, i] = (F_plus - F_moins) / (2 * h)            # On applique les différences finies centrées sur chaque colonne

    return J



def NewtonRn(F,X0,precision=1e-6, n=200):
    X = X0
    compteur = 0
    norme = np.linalg.norm(F(X))                          # Ici, on essaie de trouver X tel que la norme euclidienne de "F(X)" est majorée par la précision
    while norme > precision and compteur < n:   
        jac = approx_jacob(F,X)                           # Calcul de la jacobienne en X_i
        delta = np.linalg.solve(jac, -F(X))               # Calcul de la solution de jac*Y = -F(X), qui est la valeur de X_i+1
        X += delta                                        # On utilise l'équation donné liant X_i et X_i+1
        compteur += 1
        norme = np.linalg.norm(F(X))
    if norme > precision:                                 # Cas où la norme n'est jamais majorée par la précision 
        print("L'algorithme ne converge pas en " + str(n) +" étapes")
        return False
    print(compteur)
    return X







#Second programme pour la jacobienne. Les deux fonctionnent, c'est juste au cas où




def jacobian(f, x, h=1e-12):                      # On code la jacobienne par différence finie
    n = len(x)                                    # Dimension du vecteur X
    jac = [[0] * n for _ in range(n)]             # On dimensionne la jacobienne en la remplissant de 0
    fx = f(x)                                     # Ici, on stock la valeur de f (car on va la faire varier)
    for i in range(n):                            # On code f(x+h) pour toutes les fonctions de F
        x_i = x[i]
        x[i] = x_i + h
        fx_i = f(x)
        for j in range(n):
            jac[j][i] = (fx_i[j] - fx[j]) / h     # On utilise la différence finie (f(x+h)-f(x))/h et on le stock dans la jacobienne
        x[i] = x_i
    return jac




def NewtonRn2(F,X0,precision = 1e-8):                      # Pour lexemple P = [p1,p2,p3] surtout
    X = X0
    dim = len(X0) 
    for i in range(20):                                    #'20 est arbitraire (marche bien pour lexemple P dans R3)'
        J0 = jacobian(F,X0)                                #'On enregistre la jacobienne de F en X0'
        delta = - np.dot((np.linalg.inv(J0)),F(X0))        #'np.dot permet davoir le produit des matrices (J0)^(-1) et F(X0)'
        X = X + delta                                      #'On utilise la relation donnée'
        if np.linalg.norm((X,X0)) <= precision:            #'Si la norme est majorée, alors on a trouvé une racine'
            return X
        X0 = X                                             #'Sinon, on met à jour X et on recommence'
    return X , F(X)








"---------------------------------------- Tests de Newton-Raphson en dimension n ---------------------------------------------------------------------"



#Tests de NewtonRn


def h1(x):
    return np.array([10*(x[1] - x[0]**2), 1 - x[0]])

def h2(x):
    return np.array([x[0]**2 - x[1], 2*x[1] - 4])

def h3(x):
    return np.array([4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
                     2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)])

def h4(x):
    return np.array([(x[0] + 2*x[1] - 7)**2, (2*x[0] + x[1] - 5)**2])


def h5(x):
    return np.array([x[0] + 10*x[1]-5, np.sqrt(5)*(x[2] - x[3]), (x[1] - 2*x[2])**2, np.sqrt(10)*(x[0] - x[3])**2])

def h6(x):
    return np.array([x[1]**2+4*np.cos(x[0]), np.exp(x[0])-np.power(x[1], 1.5)])

#print(NewtonRn(h1, [2,-2]))
#print(NewtonRn(h2, [20,-20]))
#print(NewtonRn(h3, [20,-20]))
#print(NewtonRn(h4, [20,-20]))
#print(NewtonRn(h5, [20,-20,20,21]))




"-------------------"

#Test de NewtonRn2


def p1(x):
    return x[0] + x[1] - x[2]

def p2(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.exp(x[2]) - np.exp(6)

def p3(x):
    return x[0] * x[1] * x[2] - 6


def P(x):                           
    return [p1(x), p2(x), p3(x)]




Y1 = [1,1,1]
'[0.17173149 5.82561765 5.99734914]' # Racine trouvée pour le point initial Y1 pour l'algorithme de NewtonRn2
Y2 = [2,2,2]
'[5.82699503 0.1716537  5.99864873]' # Racine trouvée pour le point initial Y1 pour l'algorithme de NewtonRn2


# On peut remarquer que les solutions sont quasiment les même en interversant les coordonnées, ça peut s'expliquer par le fait que p1 et p3 sont symétriques par rapport à x et y

J1 = jacobian(P, Y1)
J2 = jacobian(P,Y2)


#print("Le X recherché est: " + str(NewtonRn2(P,Y1)[0]))
#print("La fonction vaut alors: " + str(NewtonRn2(P,Y1)[1]))

#print("Le X recherché est: " + str(NewtonRn2(P,Y2)[0]))
#print("La fonction vaut alors: " + str(NewtonRn2(P,Y2)[1]))











"----------------------------------------------------------------------------------------------------------------------------------------"
"---------------------------------------- Algorithme pour fractales ---------------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------------------------"





def df(f, x, h=1e-6):                  # On utilise la différence finie pour trouver la dérivée, à la fois pour le faire pour toute fonction et pour rester cohérent avec la partie d'avant
    return((f(x+h)-f(x-h))/(2*h))




def newton(z,f):                       # On redéfinit NewtonR1 avec df dans le cas complexe
    for i in range(20):                # 20 est encore arbitraire, on l'utilise car ça marche bien avec nos exemples en bas
        z = z - f(z) / df(f,z)  
    return z




def fractale(g,resolution=500):
    X = np.linspace(-2, 2, resolution)          # On crée un tableau de points régulièrement espacé pour X et Y
    Y = np.linspace(-2, 2, resolution)
    x, y = np.meshgrid(X, Y)                    # On maille l'ensemble X et Y (ici, [-2,2]x[-2,2])
    z = x + 1j*y                                # On utilise la corrélation R2 <---> C 
    
    w = np.zeros_like(z)                        # On remplit un array avec la même taille que la grille
    for i in range(len(z)):                     # Les 2 "range" permettent de visiter tous les points de w
        for j in range(len(z[0])):
            w[i][j] = newton(z[i][j],g)         # Pour chaque point de la grille, on associe la valeur dans l'array
    plt.imshow(np.angle(w))                     # Enfin, on utilise les angles afin de "colorier" le trajet de tous les z
    plt.gca().invert_yaxis()                    # Bug de matplolib, on inverse l'axe des ordonnées pour avoir un repère cohérent
    plt.show()




"---------------------------------------- Tests de l'algorithme pour les fractales ---------------------------------------------------------------------"


#Test des fractales



def t1(z):
    return 2*z**3 -2

def t2(z):
    return -z**3 + z**2 + 14

def t3(z):
    return 9*z**4 + 10

def t4(z):
    return z**5 - 1



# On remarque que pour t5 et t6, les fractales obtenues sont non usuelles

def t5(z):
    return z**6 + 6*z**3 - 1

def t6(z):
    return z**3 + z**2 + 14



#fractale(t1)
#fractale(t2)
#fractale(t3)
#fractale(t4)

#fractale(t5)
#fractale(t6)
