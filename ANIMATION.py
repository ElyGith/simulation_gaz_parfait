#@

# Une intro sur la loi
# L'origine microscopiquede la loi (expliquer  quoi )
# Expliquer comment à partir du micro on passe au macro
# On peut en expliquant mettre les lignes de code qui étayent nos propos
# Le but de ce code (important)
# Toutes ces équations seront accompagnées par les graphes
# mettre une conclusion
# REDIGER EN LATEK en créant un compte sur OverLeaf



# Les importations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
                                     #ANIMATION 
# =============================================================================

L = 10  # Taille de la boite carrée

N = 10  # Nombre de particules

dt = 0.09  # pas de temps

# Crée un tableau de vitesses initiales aléatoires(v_x et v_y)
vitesse = np.random.normal(0, 1, size=(N, 2))


# crée un tableau de N valeurs identiques toutes égales à 0.5 (rayon de toutes les particules)
rayon = np.full(N, 0.5)

masse = np.ones(N)  # Crée un tableau de N masses égales égales à 1

dimension = 2

# Tableau des positions des particules(Toutes les valeurs initialisées à 0)
X = np.zeros((N, dimension))

# pour éviter que les particules se chevauchent à l'initialisation.
for i in range(N):
    while True:
        # Tire une position aléatoire pour la particule i de sorte qu'elle soit entièrement dans la boite
        nouvelle_position = np.random.uniform(rayon[i], L - rayon[i], size=2)

        # On suppose que cette position ne chevauche aucune particule déjà placée
        chevauchement = False

        for j in range(i):

            # si la distance entre les particules i et j est inférieure à la somme des rayons....
            if np.linalg.norm(nouvelle_position - X[j]) < 2*rayon[j]:

                chevauchement = True  # ......les deux particules se chevauchent

                break

        if not chevauchement:  # si il n'y a pas eu chevauchement.....
            # ......on enregistre la position pour la particule i
            X[i] = nouvelle_position
            break

# Arrondit toutes les positions à 3 décimales pour éviter les nombres trop longs
X = np.round(X, 3)


# Fonction de collision entre particules
def collision(i, j):
    vitesse_relative = vitesse[i] - vitesse[j]

    # Vecteur unitaire qui pointe de j vers i (c'est la direction du choc)
    centre_normalise = (X[i]-X[j])/np.linalg.norm(X[i] - X[j])

    # on projette vitesse_relative sur la direction du choc pour savoir si les particules se rapprochent ou s'éloignent
    vitesse_relative_normale = np.dot(vitesse_relative, centre_normalise)

    if vitesse_relative_normale >= 0:  # si les deux particules s'éloignent on sort de la fonction

        return

    # quantité de mouvement transmise d'une particule àl'autre
    q = -2*np.dot(vitesse_relative, centre_normalise) * \
        centre_normalise/(1/masse[i] + 1/masse[j])

    vitesse[i] = vitesse[i] + (q/masse[i])  # on recalcule les vitesses
    vitesse[j] = vitesse[j] - (q/masse[j])


# Affichage initial et calcul énergie
E = 0
for i in range(N):
    E += 1/2*masse[i]*np.linalg.norm(vitesse[i])**2
# permet de vérifier que l'énergie est conservée pendant la simulation.
print('energie cinétique initiale du système :', E)


# MÉTHODE 1 : Simulation pas-à-pas

# Varaibles pour les mesures

kB = 1.0  # Constante de Boltzmann prise égale à 1. Cela simplifie les calculs et n'affecte pas la validité des lois, car nous vérifions uniquement des relations de proportionalité. Et par souci d'uniformité des unités de mesure.

somme_impulsions = 0.0  # somme des impulsions sur le mur de droite.

# Utile plus tard pour calculer la pression sur le mur de droite. En effet à l'équilibre la pression dans toute la boite est la meme quelque soit le mur. Donc, nous prenons le mur de droite car c'est plus simple à calculer !

nb_collisions_mur = 0  # nombre de collisions avec le mur de droite

temps_total = 0.0  # Temps total de simulation

# Listes pour stocker les valeurs (pour les graphiques plus tard)
liste_temps = []  # STockera les instants t où on effectuera une mesure

liste_T = []  # Stockera les températures mesurées

liste_P = []  # Stockera les pressions mesurées

liste_E = []  # Stockera les énergies cinétiques mesurées

# Calcul de l'nergie cinétique
E_init = 0.0

for k in range(N):
    # Energie cinétique totale initiale
    E_init += 0.5 * masse[k] * (vitesse[k, 0]**2 + vitesse[k, 1]**2)

print(f"Énergie initiale du système : {E_init:.6f}")


plt.figure(figsize =(8,8))
plt.ion()


for iteration in range(100):  # Collision avec les murs

    # Met à jour toutes les positions des particules(cette ligne peut créer des chevauchements de particules)
    X = X + vitesse*dt

    temps_total += dt

    for i in range(N):
        if X[i, 0] - rayon[i] <= 0 or X[i, 0] + rayon[i] >= L:
            if X[i, 0] + rayon[i] >= L:  # pour effectuer les mesures sur le mur de droite
                impulsion = 2 * masse[i] * abs(vitesse[i, 0])
                somme_impulsions += impulsion
                nb_collisions_mur += 1
            vitesse[i, 0] *= -1

        if X[i, 1] - rayon[i] <= 0 or X[i, 1] + rayon[i] >= L:  # idem pour les murs horizontaux
            vitesse[i, 1] *= -1

        for j in range(i+1, N):  # Collision entre particules

            dX = X[i] - X[j]  # Calcule le vecteur qui va de j vers i

            # distance entre les centres des deux particules
            dist = np.linalg.norm(dX)

            dist_collision = rayon[i] + rayon[j]

            if dist > 0 and dist <= dist_collision:  # si collision détectée.....

                # On vérifie d'abord si les partiucles se chevauchent

                normale = dX / dist  # la direction du choc

                overlap = dist_collision - dist  # Calcul de combien les particules se chevauchennt

                # on décale la particule i dans la direction du vecteur "normale"
                X[i] += normale * overlap / 2

                # on décale j dans le sens inverse. Les particules ne se chevauchent donc plus, elles se touchent juste.
                X[j] -= normale * overlap / 2

                # Application du choc (les particules sont maintenant au contact)
                collision(i, j)

    if iteration % 10 == 0:
        v2_moyen = np.mean(vitesse[:, 0]**2 + vitesse[:, 1]**2)
        # Calcul de la température du système
        T = (masse[0] * v2_moyen) / (2 * kB)
        # Energie cinétique totale du système
        E = 0.5 * np.sum(masse * (vitesse[:, 0]**2 + vitesse[:, 1]**2))
        if temps_total > 0:
            # Pression totale dans la boite
            P = somme_impulsions / (temps_total * L)
        else:
            P = 0.0
        liste_temps.append(temps_total)
        liste_T.append(T)
        liste_P.append(P)
        liste_E.append(E)
        print(f"t={temps_total:.2f}s | T={T:.4f} | P={P:.4f} | E={E:.6f}")
    
    # animation de la siumulation
    plt.clf()                                      # Efface le graphique précédent
    ax = plt.gca()                                 # Récupère les axes actuels
    
    for k in range(N):                             # Dessine chaque particule
        disque = Circle((X[k,0], X[k,1]), rayon[k], fill=True, alpha=0.7, color='royalblue', edgecolor='black')
        ax.add_patch(disque)
    
    plt.scatter(X[:,0], X[:,1], s=20, color='darkblue', zorder=3)  # Petit point au centre des particules
    
    plt.xlim(0, L)                                 # Limite axe X
    plt.ylim(0, L)                                 # Limite axe Y
    ax.set_aspect("equal", adjustable="box")       # Pour que les cercles soient ronds
    plt.title(f"Simulation Gaz Parfaits - Itération {iteration}")  # Titre avec numéro d'itération
    plt.pause(0.005)  # Pause pour créer l'animation
       


# Résultats finaux

print("\n" + "="*50)
print("RÉSULTATS FINAUX - MÉTHODE 1")
print("="*50)

print(f"Temps total simulé : {temps_total:.4f} secondes")
print(f"Nombre de collisions avec le mur de droite : {nb_collisions_mur}")

if temps_total > 0:
    P_finale = somme_impulsions / (temps_total * L)
    print(f"Pression moyenne sur le mur de droite : {P_finale:.6f}")
else:
    print("Pression non calculée (temps total nul)")

print(f"Énergie initiale : {E_init:.6f}")

if liste_E:
    print(f"Énergie finale : {liste_E[-1]:.6f}")
    variation = abs(liste_E[-1] - E_init)
    print(f"Variation d'énergie : {variation:.2e}")
    if variation < 1e-10:
        print("✅ Énergie conservée (variation < 1e-10)")
    else:
        print("⚠️ Attention : l'énergie n'est pas parfaitement conservée")
else:
    print("Aucune mesure d'énergie enregistrée")
    


#@
