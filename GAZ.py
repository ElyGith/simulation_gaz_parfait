# Les importations
import numpy as np 
import heapq
import matplotlib.pyplot as plt

#Initialiser les vitesse avec loi normale et enlever la moyenne pour éviter qu'il y ait un drift et que la moyenne des particules soit centrées en 0 
#Dans la méthode 1, creer un truc qui calcule le nombre de collisiosn pour avoir la pression macroscopique , et la température
#Faire un graphe de P en fonction de T, L fixé et P en fonction de 1/V 
#Pour L plus grand, il faut augmenter le nombre de pas
#Un graphe de P en fonction de N aussi. Et tout ces graphes doit etre linéaire 
#Vérifier à chaque fois que Energie cinétique est cste
#Une intro sur la loi 
#L'origine microscopiquede la loi (expliquer  quoi )
#Expliquer comment à partir du micro on passe au macro 
#On peut en expliquant mettre les lignes de code qui étayent nos propos 
#Le but de ce code (important) 
#Toutes ces équations seront accompagnées par les graphes
#mettre une conclusion 
#REDIGER EN LATEK en créant un compte sur OverLeaf 




# Constantes et initialisation

L = 10   #Taille de la boite carrée

N = 10   #Nombre de particules

dt = 0.0001  #pas de temps

vitesse = np.random.normal(0, 1, size=(N,2))  #Crée un tableau de vitesses initiales aléatoires(v_x et v_y)


rayon = np.full(N, 0.5) #crée un tableau de N valeurs identiques toutes égales à 0.5 (rayon de toutes les particules)

masse = np.ones(N)  #Crée un tableau de N masses égales égales à 1 

dimension = 2

X = np.zeros((N,dimension)) #Tableau des positions des particules(Toutes les valeurs initialisées à 0)

for i in range(N) : #pour éviter que les particules se chevauchent à l'initialisation.
  while True : 
    nouvelle_position = np.random.uniform(rayon[i], L - rayon[i], size = 2) #Tire une position aléatoire pour la particule i de sorte qu'elle soit entièrement dans la boite
    
    chevauchement = False #On suppose que cette position ne chevauche aucune particule déjà placée
    
    for j in range(i) : 
      
      if np.linalg.norm(nouvelle_position - X[j]) < 2*rayon[j] : #si la distance entre les particules i et j est inférieure à la somme des rayons....
        
        chevauchement = True #......les deux particules se chevauchent
        
        break
      
    if not chevauchement : # si il n'y a pas eu chevauchement.....
        X[i] = nouvelle_position # ......on enregistre la position pour la particule i 
        break 
    
X = np.round(X, 3)   #Arrondit toutes les positions à 3 décimales pour éviter les nombres trop longs




# Fonction de collision entre particules
def collision(i, j):
    vitesse_relative = vitesse[i] - vitesse[j] 
    
    centre_normalise = (X[i]-X[j])/np.linalg.norm(X[i] -X[j]) #Vecteur unitaire qui pointe de j vers i (c'est la direction du choc)
    
    vitesse_relative_normale = np.dot(vitesse_relative, centre_normalise) #on projette vitesse_relative sur la direction du choc pour savoir si les particules se rapprochent ou s'éloignent 
    
    if vitesse_relative_normale >= 0: #si les deux particules s'éloignent on sort de la fonction
      
        return
      
    q = -2*np.dot(vitesse_relative, centre_normalise)*centre_normalise/(1/masse[i] + 1/masse[j]) #quantité de mouvement transmise d'une particule àl'autre
    
    vitesse[i] = vitesse[i] + (q/masse[i]) #on recalcule les vitesses 
    vitesse[j] = vitesse[j] - (q/masse[j])
    
    

# Affichage initial et calcul énergie
E = 0
for i in range(N):
    E += 1/2*masse[i]*np.linalg.norm(vitesse[i])**2
print('energie cinétique initiale du système :', E) # permet de vérifier que l'énergie est conservée pendant la simulation.



# MÉTHODE 1 : Simulation pas-à-pas

for iteration in range(40): #Collision avec les murs 
  
    X = X + vitesse*dt # Met à jour toutes les positions des particules(cette ligne peut créer des chevauchements de particules)
    
    plt.scatter(X[:,0], X[:,1]) 
    
    for i in range(N):
      
        if X[i, 0] - rayon[i] <= 0 or X[i, 0] + rayon[i] >= L: #si la particule rencontre un mur vertical(gauche ou droit)....
          
            vitesse[i, 0] *= -1 #....il ya rebond (on inverse sa vitesse)
            
        if X[i, 1] - rayon[i] <= 0 or X[i, 1] + rayon[i] >= L: #idem pour les murs horizontaux
          
            vitesse[i, 1] *= -1
            
        for j in range(i+1, N):  #Collision entre particules
            
          
            dX = X[i] - X[j] #Calcule le vecteur qui va de j vers i 
            
            dist = np.linalg.norm(dX) # distance entre les centres des deux particules
            
            dist_collision = rayon[i] + rayon[j]
            
            if dist > 0 and dist <= dist_collision: #si collision détectée.....
              
                #On vérifie d'abord si les partiucles se chevauchent 
                
                normale = dX / dist #la direction du choc
                
                overlap = dist_collision - dist # Calcul de combien les particules se chevauchennt
                
                X[i] += normale * overlap / 2 # on décale la particule i dans la direction du vecteur "normale"
                
                X[j] -= normale * overlap / 2 # on décale j dans le sens inverse. Les particules ne se chevauchent donc plus, elles se touchent juste.  
                
                collision(i, j) #Application du choc (les particules sont maintenant au contact)
                
                

# Fonction pour MÉTHODE 2 : temps avant collision entre deux particules i et j 
def collision_time(i, j):
  
    r = X[i] - X[j] #position relative de i par rapport à j 
    
    v = vitesse[i] - vitesse[j] # vitesse relative de i par rapport à j
    
    R = rayon[i] + rayon[j]
    
    # On veut trouver t (temps) où la distance entre les deux particules est exactement R
    
    # La distance au carré après un temps t s'écrit :  distance²(t) = ||r + v*t||² = (v·v)*t² + 2*(r·v)*t + (r·r)
    
    # On veut : distance²(t) = R², donc : (v·v)*t² + 2*(r·v)*t + (r·r - R²) = 0 . Nous allons donc résoudre cette équation
    
    a = np.dot(v, v)
    
    b = 2 * np.dot(r, v)
    
    c = np.dot(r, r) - R**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None # Les particules ne se rencontrent jamais 
      
    if a == 0: #Si la vitesse relative est nulle c'-à-d si les particules vont exactement à la meme vitesse
        return -c/b if b != 0 else None
      
    t1 = (-b - np.sqrt(discriminant)) / (2*a) 
    
    t2 = (-b + np.sqrt(discriminant)) / (2*a) 
    
    # On retourne le temps (positif donc collision future) avant la prochaine collision 
    
    # Positif car si t = 0 les particules sont déjà en contact ou se chevauchent et si t < 0 la collision a déjà eu lieu 
    
    t = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else None
    return t



# Fonction pour MÉTHODE 2 : temps avant collision avec un mur
def collision_time_mur(i):
  
    times = [] # Liste qui va stocker les temps de collision avec chaque mur (gauche, droite, bas, haut)
    
    # Collision avec le mur de gauche (x=0): 
    
    if vitesse[i, 0] < 0: # Si la particule va vers le mur de gauche
        t_left = (rayon[i] - X[i, 0]) / vitesse[i, 0] # On calcule le temps auquel il y aura collision 
        if t_left > 0:
            times.append(t_left)
            
    
    # Collision avec le mur de droite (x= L) : meme processus que précédemment 
    if vitesse[i, 0] > 0:
        t_right = (L - rayon[i] - X[i, 0]) / vitesse[i, 0]
        if t_right > 0:
            times.append(t_right)
            
    # Collision avec le mur du bas(y= 0) : 
    if vitesse[i, 1] < 0:
        t_bottom = (rayon[i] - X[i, 1]) / vitesse[i, 1]
        if t_bottom > 0:
            times.append(t_bottom)
            
    # Collision avec le mur du haut (y=L) :
    if vitesse[i, 1] > 0:
        t_top = (L - rayon[i] - X[i, 1]) / vitesse[i, 1]
        if t_top > 0:
            times.append(t_top)
            
    if times: # Si times est non vide.....
        return min(times) # On retourne la plus petite valeur c'à-d la première collision future de la particule
      
    return None # Si times est vide la particule ne touchera aucun mur 
  

# MÉTHODE 2 : initialisation de la file de priorité des évènements 
events = [] # Liste qui va contenir tous les évènements futurs (collisions entre particules ou avec les murs). Chaque évènement sera un couple (temps, (particule1, particule2))

def init_events():
    events.clear() #On vide la liste avant de la remplir
    for i in range(N):
        #Collisions avec les murs :
        t_mur = collision_time_mur(i) # Calcule le temps avant que la particule i ne touche le mur le plus proche
        
        if t_mur is not None: # Si une collision est possible....
            heapq.heappush(events, (t_mur, (i, None))) # Ajoute l'évènement dans la file
            
        #Collisions entre particules :
        for j in range(i+1, N):
            t = collision_time(i, j) #Calcule le temps avant que i et j  ne se touchent
            
            # Idem pour la collision avec le mur 
            if t is not None: 
                print(f"Collision entre particules {i} et {j} dans {t:.2f} secondes.")
                heapq.heappush(events, (t, (i, j)))
                
                
init_events()

# MÉTHODE 2 : boucle principale événementielle
NB_ITER = 15 # Nombre maximal d'itérations
INIT = False
while NB_ITER > 0:
  
    NB_ITER -= 1 #Diminue le compteur d'itérations à chaque tour
    
    if not events: #Si la liste events est vide (plus aucun évènement à venir) on sort de la boucle.
        break
      
    plt.scatter(X[:,0], X[:,1])
    
    t, (a, b) = heapq.heappop(events) #Extrait l'évènement avec le plus petit temps de la file 
    if t < 1e-10:
        continue
    for k in range(N):
      
        X[k] += vitesse[k] * t #On avance les particules jusqu'au moment de la collision
        
    if b is None: # Si collision avec un mur, on inverse la vitesse de la particule
        vitesse[a] *= -1 #erreur ca marche pas quand vitesse fait angle avec le mur 
        
    else: #Sinon, on met à jour leurs vitesses
        collision(a, b)
        
    if INIT == True:
        pass
      
    # Après avoir extrait la collision la plus proche, toutes les positions sont recalculées donc les evènements restant dans la file sont obsolètes (ils correspondent à d'anciennes positions et vitesses). Alors on vide la file des évènements obsolètes et on la remplit à nouveau avec le calcul des nouvelles collisions futures
    else:
        events.clear() 
        init_events()
        
        
        




