#@
# Les importations
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
                                        #TRACAGE DE GRAPHES
# =============================================================================
def simuler(L, N,E_cible, NB_ITER =None ,dt = 0.01, rayon = 0.00000000001 ,masse = 1.0):   # Rayon très petit pour imiter la strucure ponctuelle des particules du gaz parfait !  
    if NB_ITER is None:
       NB_ITER = int(500 * (L / 10) ** 2 * (15 / N)) #plus L est grand, plus on itère
    #nous allons lancer une simulation et retourner la pression mesurée, ainsi que la température finale
    #L : taille du coté de la boite carré 
    
    #N : nombre de particules 
    
    #E_cible : Energie cinétique totale imposée au depart (utile pour les simulations car vitesses générées aléatoirement à chaque simulation implique :
        # énergies cinétiques aléatoires et donc
        # T ou N aléatoires (car E=kb*N*T)
        #Si on veut tracer P avec N ou T fixés à chaque simulation il y a un problème car T ou N vont toujours varier   
    
    #NB_ITER : nombre de pas 
    # nous llons retourner P et T 
    kb = 1.0
    rayons =  np.full(N,rayon) # constante de Boltzmann


    # initialisation des positions sans chevauchement
    X = np.zeros((N,2))       # tableau des positions, initialisé à zéro
    X = np.zeros((N, 2))
    for i in range(N):
        compteur = 0 # compte le nombre de tentative de placement 
        while True:
            pos = np.random.uniform(rayons[i], L - rayons[i], size=2)#tire une position aléatoire entierement à l'interieur de la boite
            ok = all(np.linalg.norm(pos - X[j]) >= 2 * rayon for j in range(i)) #vérifions qu'elle ne chevauche aucune particule deja placé
            if ok:
                X[i] = pos
                break# la position est bonne 
            compteur += 1
            if compteur > 10000:   # si après 10000 essaie on a pas trouvé de place , la boite est trop petite 
                print(f"⚠️ IMPOSSIBLE de placer la particule {i} dans L={L}")
                break

    #initialisation des vitesses             
    vit = np.random.normal(0,1,size = (N,2)) #vitesses aléatoire
    vit -= vit.mean(axis =0) #supprime le drift
    

    E_actuelle = 0.5*masse* np.sum(vit**2) # Energie cinétique actuelle
    vit *= np.sqrt(E_cible / E_actuelle) # recalibrer la vitesse aléatoire pour obtenir E_cible pour rester conforme au conditions initiales
    # fonction de collision elastique entre deux particules i et j
    def collision(i, j):
       vrel = vit[i] - vit[j]
       n_hat = (X[i] - X[j]) / np.linalg.norm(X[i] - X[j]) # vecteur unitaire j -> i 
       if np.dot(vrel , n_hat) >= 0 : #particule s'eloigne: rien à faire 
           return
       #quantité de mouvement échangé lors du choc élastique
       q = -2*np.dot(vrel, n_hat)*n_hat / (1/masses[i] + 1 / masses[j])
       vit[i] += q / masses[i]
       vit[j] -= q / masses[j]
       
    # boucle principale de la simulation 
    somme_impulsions = 0.0
    temps_total = 0.0
    
    for _ in range(NB_ITER):
        X += vit * dt # avance les positions d'un pas de temps 
        temps_total += dt # incrémente le temps total          
        for i in range(N):
            
            # rebond sur le mur de droite ( x = L )
            if X[i,0] + rayons[i] >= L :
                somme_impulsions += 2*masses[i] *abs(vit[i,0])
                vit[i,0] *= -1 
            # rebond sur le mur de gauche (x = 0)
            elif X[i,0] - rayons[i] <= 0:
                somme_impulsions += 2 * masses[i] * abs(vit[i, 0])
                vit[i,0] *= -1
            # rebond sur le mur du haut (y = L) 
            if X[i, 1] + rayons[i] >= L:
                somme_impulsions += 2 * masses[i] * abs(vit[i, 1])
                vit[i, 1] *= -1
            #rebond sur le mur du bas (y =0)
            elif X[i,1] - rayons[i] <= 0:
                somme_impulsions += 2 * masses[i] * abs(vit[i, 1])
                vit[i , 1] *= -1
            
            # collision entre les particules 
            
            for j in range ( i + 1, N):
                dX = X[i] - X[j] 
                dist = np.linalg.norm(dX)
                d_col = rayons[i] + rayons[j]
                
                if 0 < dist <= d_col:
                    #On regarde si les particules se chevauchent déjà au moment de la détection
                    n_hat  =  dX / dist #vecteur unitaire de j vers i 
                    overlap = d_col - dist #de combien elle se chevauchent
                    X[i] += n_hat * overlap /2 #On décale i dans la direction du vecteur dX
                    X[j] -= n_hat*overlap /2 #On décale j dans la direction opposée
                    collision(i,j)
                    
    #calcul des grandeur macroscopiques en fin de simulation
    P = somme_impulsions / (temps_total * 4 * L) # pression
    v2m = np.mean(vit[:, 0] ** 2 + vit[:,1] ** 2)# vitesse quadratique moyenne.
    T = (masse* v2m) / (2 * kb) #température
    
    return P,T


#@
#GRAPHE 1 : P en fonction de 1/V (N et T fixés)
print("=" * 55)
print("GRAPHE 1 : P = f(1/V) - N et T fixés")
print("="* 55)
N_fixe = 15 #nombre de particule
E_fixe = 20.0 # energie cinétique fixés donc T fixés
kb = 1.0

liste_L_V = [10,12,15,18,22,27,33] # pour les differentes tailles de boite on prends differents volumes 
liste_p_V = [] # on stockera les pressions mesurées
liste_invV = [] # on stockera les 1/V 
# la simulation est aléatoire , les positions et les vitesses initiales sont tirées au hasard
#donc deux runs aves les mmêmes paramètres(meme L, meme , N meme E) donnent des pressions 
#légèrement différentes. un seul run peut donc donner un point aberrant par malchance 
#pour corriger ca, on lance NB_RUNS simulations pour chaque point et on moyenne les pressions obtenues
# les fluctuations se compensent et le resultat est bien plus stable.
for L in  liste_L_V:
    NB_RUNS = 1 # nombres de simulations qu'on va lancer pour ce L
    pressions = [] # liste qui va stocker les 5 pressions mesurées 
    for _ in range(NB_RUNS): #on lances les 5 simulations
        P,T= simuler(L=L, N=N_fixe, E_cible=E_fixe) # une simulation 
        pressions.append(P) # on stock la pression obtenue
    P_moyen =np.mean(pressions) # on calcule la moyenne des 5 pressions 
    V = L **2 # on calcul le volume qui correspond à l'aire en dimension 2 
    liste_p_V.append(P_moyen) # on stock la pression moyenne 
    liste_invV.append(1 / V) # on stock 1/V 
    print(f"L={L:4} V={V:5} 1/V={1/V :.6f} P={P_moyen:.5f} T={T:.3f}")
    '''
    P,T = simuler(L=L , N = N_fixe,E_cible=E_fixe) #lance la simulation pour ce L 
    V = L**2
    liste_p_V.append(P)
    liste_invV.append(1/V)
    print(f"L={L:4} V={V:5} 1/V={1/V:.6f} P={P:.5f} T={T:.3f}")
#droite théorique de P en fonctyion de 1/V'''
T_theo = E_fixe / (N_fixe * kb)  # temperature théorique à partir de  E = N·kB·T
pente_theo_V = N_fixe * kb * T_theo
X_fit = np.linspace(0,max(liste_invV) * 1.15 ,200)
liste_V = [1 / invV for invV in liste_invV]

X_fit_Volume = np.linspace(min(liste_V),max(liste_V) * 1.15 ,200)


plt.figure(figsize = (7,5))
plt.plot(liste_invV, liste_p_V, 'o', color = 'steelblue',markersize = 8 , label = 'simulation')
plt.plot(X_fit, pente_theo_V * X_fit, '--', color ='tomato', label = f'Théorie  P = NkT/V (pente = {pente_theo_V:.2f})')
plt.xlabel('1/V (= 1/L²)')
plt.ylabel('pression P')
plt.title(f'P en fonction de 1/V\n(N = {N_fixe}, E = {E_fixe})')
plt.legend()
plt.grid(True, linestyle = '--',alpha = 0.5)
plt.show()
print()    

plt.figure(figsize = (7,5))
plt.plot(liste_V, liste_p_V, 'o', color = 'steelblue',markersize = 8 , label = 'simulation')
plt.plot(X_fit_Volume, pente_theo_V/X_fit_Volume, '--', color ='tomato', label = f'Théorie  P = NkT/V (pente = {pente_theo_V:.2f})')
plt.xlabel('V)')
plt.ylabel('pression P')
plt.title(f'P en fonction de V\n(N = {N_fixe}, E = {E_fixe})')
plt.legend()
plt.grid(True, linestyle = '--',alpha = 0.5)
plt.show()
print()    

#@

#GRAPHE 2 : P en fonction de T (N et L fixés)

print("=" * 55)
print("GRAPHE 2 : P = f(T) - L et N fixés")
print("="* 55)
L_fixe = 12 #taille de la boite 
N_fixe_T = 20 # nombre de particules

# différentes énergies → différentes températures
# T = E /(N*kB) donc E = N*kB*T
# on choisit des T cibles puis on en déduit E_cible
liste_T_cibles = [1.0,2.0,4.0,6.0,9.0,12.0,16.0,20.0] # pour les differentes tailles de boite on prends differents volumes 
liste_p_T = [] # on stockera les pressions mesurées
liste_T_mes = [] # on stockera les températures mesurées 
# on fait les 5 sumulations comme precedement et on prends la moyenne des 5 pressions 

for T_cible in liste_T_cibles:
   E_cible = N_fixe_T*kb *T_cible # energie correspondant à cette température cible
   NB_RUNS = 1
   pressions = []
   for _ in range(NB_RUNS):
       P,T = simuler(L = L_fixe, N = N_fixe_T,E_cible=E_cible)
       pressions.append(P)
   P_moyen = np.mean(pressions)
   liste_p_T.append(P_moyen)
   liste_T_mes.append(T)
   print(f"T_cible={T_cible:5.1f} T_mesuré={T:.3f} P={P_moyen:.5f}")
   
#droite théorique P = (N*kB / V) * T  -> pente = N*kB / V
V_fixe = L_fixe ** 2 #volume de la boite 
pente_theo_T = N_fixe_T * kb / V_fixe
X_fit = np.linspace(0,max(liste_T_mes) * 1.15 ,200)

plt.figure(figsize = (7,5))
plt.plot(liste_T_mes, liste_p_T, 'o', color = 'darkorange',markersize = 8 , label = 'simulation')
plt.plot(X_fit, pente_theo_T * X_fit, '--', color ='steelblue', label = f'Théorie  P = NkT/V (pente = {pente_theo_T:.4f})')
plt.xlabel('Température  (= E / Nkb)')
plt.ylabel('pression P')
plt.title(f'P en fonction de T\n(L = {L_fixe}, N = {N_fixe_T})')
plt.legend()
plt.grid(True, linestyle = '--',alpha = 0.5)
plt.tight_layout()
plt.show()
print()

#@

#GRAPHE 3 : P en fonction de N (T et L fixés)

print("=" * 55)
print("GRAPHE 3 : P = f(N) - L et T fixés")
print("="* 55)
L_fixe_N = 15 #taille de boite 
T_fixe = 5 # température

liste_N = [5,8,10,13,16,20,25,30] # pour les differentes nombres de particules
liste_p_N = [] # on stockera les pressions mesurées

for N in liste_N:
    E_cible = N * kb * T_fixe # énergie adapté pour garder T fixée quel que soit 
    NB_RUNS = 1
    pressions = []
    for _ in range(NB_RUNS):
        P, T = simuler(L=L_fixe_N, N=N, E_cible=E_cible)
        pressions.append(P)
    P_moyen = np.mean(pressions)
    liste_p_N.append(P_moyen)
    print(f"N={N:3} E_cible={E_cible:6.1f} T_mesuré={T:.3f} P={P_moyen:.5f}")
   
# droite théorique P = (kB*T / V) × N  → pente = kB*T / V
V_fixe_N = L_fixe_N ** 2 #volume de la boite 
pente_theo_N = T_fixe * kb / V_fixe_N
X_fit_N = np.linspace(0,max(liste_N) * 1.15 ,200)

plt.figure(figsize = (7,5))
plt.plot(liste_N, liste_p_N, 'o', color = 'seagreen',markersize = 8 , label = 'simulation')
plt.plot(X_fit_N, pente_theo_N * X_fit_N ,'--', color ='crimson', label = f'Théorie  P = NkT/V (pente = {pente_theo_N:.5f})')
plt.xlabel('Nombre de particules N')
plt.ylabel('pression P')
plt.title(f'P en fonction de N\n(L = {L_fixe_N}, T = {T_fixe})')
plt.legend()
plt.grid(True, linestyle = '--',alpha = 0.5)
plt.tight_layout()
plt.show()
print()

    
    
    
    
    
    
    
    

#@
