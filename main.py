# Les importations
#@
import numpy as np
import random as rd
import time
import heapq



import matplotlib.pyplot as plt

#@
# Constante du progamme et initialisation des variables

L = 10 # Taille de la boite
N = 10 # Nombre de particule dans la boite

dt = 0.01 # delta T

vitesse = np.random.randint(0,5, size=(N,2)) # la vitesse de toute les particules

#vitesse = np.random.randn(N,2) # la vitesse de toute les particules
rayon = np.full(N, 0.5)  # rayon fixe pour toutes les particules/ permet de réduire l'erreur de conservation d'énergie
#rayon = np.random.uniform(0.1,0.5,size=N) # rayon aléatoire pour chaque particule
masse = np.random.uniform(size=N)

dimension = 2

X = np.zeros((N,dimension))



for i in range(N):
    X[i] = np.random.uniform(0,L,size = 2)

X = np.round(X,3)
#@



def collision (i, j):
    vitesse_relative = vitesse[i] - vitesse[j]
    centre_normalise = (X[i]-X[j])/np.linalg.norm(X[i] -X[j])
    vitesse_relative_normale = np.dot(vitesse_relative,centre_normalise)

    if vitesse_relative_normale >= 0: 
        return

    q = -2*np.dot(vitesse_relative,centre_normalise)*centre_normalise/(1/masse[i] + 1/masse[j]) 
    vitesse[i] = vitesse[i] + (q/masse[i])
    vitesse[j] = vitesse[j] - (q/masse[j])


#@

# plt.scatter(X[:,0],X[:,1]) # position initial 

abscisse = X[:,0]
ordonnee = X[:,1]



# Calcul de l'énergie cinétique initiale
E=0
for i in range(N):
    E += 1/2*masse[i]*np.linalg.norm(vitesse[i])**2 

print('energie cinétique initiale :', E)

###########################################################
# Simulation de la dynamique des particules : Méthode 1 
###########################################################

for i in range(40):

    X = X + vitesse*dt
   
  #  plt.pause(0.5) # quand on mets pause on affiche graphe par graphe alors que sinon on a tout d'un coup
    plt.scatter(X[:,0],X[:,1]) # position initial 

    for i in range(N):
        if X[i, 0]  -rayon[i]<= 0 or X[i, 0] + rayon[i] >= L: # abscisse qui sort du cadre
            vitesse[i, 0] *= -1
        if X[i, 1] - rayon[i] <= 0 or X[i, 1] + rayon[i] >= L: # pareil pour l'ordonnée
            vitesse[i, 1] *= -1

        for j in range(i+1,N):
            dX = X[i] - X[j] 
            dist = np.linalg.norm(dX)
            dist_collision = rayon[i] + rayon[j]

            if dist>0 and dist<=dist_collision: # inégalité psq on aura jamais d'égalité stricte
                print("Risque de COLLISIOOOON")
               # time.sleep(3)
                
                vitesse_relative = vitesse[i] - vitesse[j]
        
                centre_normalise = (X[i]-X[j])/np.linalg.norm(X[i] -X[j]) 
                centre_normalise = dX/dist

                vitesse_relative_normale = np.dot(vitesse_relative,centre_normalise)
                vitesse_relative_normale = np.dot(vitesse_relative, dX) / dist

                if vitesse_relative_normale >= 0: 
                    # pour vérifier que les particules s'éloignent pas  mais qu'elles se rapprochent et donc qu'on a une collision 
                    # soit y en a une qui se rapproche + vite que l'autre ne s'éloigne 
                    # soit les deux se rapprochent 
                    continue

                q = -2*np.dot(vitesse_relative,centre_normalise)*centre_normalise/(1/masse[i] + 1/masse[j]) # on multiplie par le centre soit ici soit dans l'ajout de la vitesse
                #q = -2*vitesse_relative_normale/(1/masse[i] + 1/masse[j]) # si on utilise celle ci faut ajouter *centre_normalise dans l'ajout de la vitesse
                vitesse[i] = vitesse[i] + (q/masse[i])
                vitesse[j] = vitesse[j] - (q/masse[j])
                print(f"Collision entre particules {i} et {j} à la position {X[i]} et {X[j]}")

                E = 0
                for k in range(N):
                    E += 1/2*masse[k]*np.linalg.norm(vitesse[k])**2 
                print('Energie cinétique après collision :', E)
 
    ### vérifier, y a surement des erreur dans le calcul des collisions, pour la vitesse relative noprmal etc.
### on perd a chaque collision de l'énergie cinétique, 
# 


#@


def collision_time(i, j):

    r = X[i] - X[j] # position  relative
    v = vitesse[i] - vitesse[j] # vitesse relative
    R = rayon[i] + rayon[j]  # distance de collision

    a = np.dot(v, v)
    b = 2 * np.dot(r, v)
    c = np.dot(r, r) - R**2

    discriminant = b**2 - 4*a*c

    print(f"Calcul de collision_time entre particules {i} et {j}: a={a}, b={b}, c={c}, discriminant={discriminant}")

    if discriminant < 0:  # pas de solution réelle ou les particules sont à la même position et ont la même vitesse
        return None  # pas de collision
    
    if a == 0:  # les particules ont la même vitesse
        return -c/b if b != 0 else None  # collision si elles sont à la distance de collision, sinon pas de collision


    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)

    # on prend le plus petit temps positif
    t = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else None

    return t

events = []


def collision_time_mur(i):

    times = []
    
    # Mur gauche (x=0)
    if vitesse[i, 0] < 0: # si son abscisse diminue
        t_left = (rayon[i] - X[i, 0]) / vitesse[i, 0]
        if t_left > 0:
            times.append(t_left)
    
    # Mur droit (x=L)
    if vitesse[i, 0] > 0: # si son abscisse augmente
        t_right = (L - rayon[i] - X[i, 0]) / vitesse[i, 0]
        if t_right > 0:
            times.append(t_right)
    
    # Mur bas (y=0)
    if vitesse[i, 1] < 0: # si son ordonnée diminue
        t_bottom = (rayon[i] - X[i, 1]) / vitesse[i, 1]
        if t_bottom > 0:
            times.append(t_bottom)
    
    # Mur haut (y=L)
    if vitesse[i, 1] > 0:  # si son ordonnée augmente
        t_top = (L - rayon[i] - X[i, 1]) / vitesse[i, 1]
        if t_top > 0:
            times.append(t_top)
    
    if times:
        return min(times)
    
    return None
       

plt.scatter(X[:,0],X[:,1]) # position initial 

for i in range(N):
    t_mur = collision_time_mur(i)
    if t_mur is not None:
        heapq.heappush(events, (t_mur, (i, None)))
    for j in range(i+1,N):
                    t = collision_time(i, j)

                    if t is not None:
                        print(f"Collision entre particules {i} et {j} dans {t:.2f} secondes.")
                        heapq.heappush(events, (t, (i, j)))


# print(heapq.heappop(events))
#@
NB_ITER = 10
while True and NB_ITER > 0:
    NB_ITER -= 1
    if not events:
        break  # sortir si plus d'événements

    # plt.pause(1) 
    plt.scatter(X[:,0],X[:,1]) # position initial 


    t, (a, b) = heapq.heappop(events)

    if t < 1e-10:  # ignorer les collisions trop proches (quasi-instantanées)
        continue

    for k in range(N):
        X[k] += vitesse[k] * t

    if b is None:  # collision avec le mur
        vitesse[a] *= -1  # inverser la vitesse de la particule
    else:  # collision entre particules
        collision(a, b)


    for j in range(N):
        if j == a or j == b:
            continue  # éviter de reprogrammer la collision actuelle

        if b is None:
            collision_time_mur_a = collision_time_mur(a)
            if collision_time_mur_a is not None and collision_time_mur_a > 1e-10:
                heapq.heappush(events, (collision_time_mur_a, (a, None)))
        else:

            ta = collision_time(a, j)
            tb = collision_time(b, j)

            collision_time_mur_a = collision_time_mur(a)
            collision_time_mur_b = collision_time_mur(b)

            if collision_time_mur_a is not None and collision_time_mur_a > 1e-10:
                heapq.heappush(events, (collision_time_mur_a, (a, None)))
            if collision_time_mur_b is not None and collision_time_mur_b > 1e-10:
                heapq.heappush(events, (collision_time_mur_b, (b, None)))

            if ta is not None and ta > 1e-10:
                print(f"Collision entre particules {a} et {j} dans {ta:.4f} secondes.")
                heapq.heappush(events, (ta, (a, j)))

            if tb is not None and tb > 1e-10:
                print(f"Collision entre particules {b} et {j} dans {tb:.4f} secondes.")
                heapq.heappush(events, (tb, (b, j)))


X[0]
X[2]

vitesse[0]

vitesse[2]
vitesse_relative = vitesse[0] - vitesse[2]

np.dot(vitesse_relative,vitesse_relative)


#@
