# resolution-d-equation-nonlineaires-sur-Rn
Projet de première année de master de mathématiques MFA

Ce projet avait pour but la résolution d'équations non-linéaires dans Rn. Plusieurs tâches m'ont été données.

Dans un premier temps, je devais examiner et implémenter deux méthodes distinctes (pour des fonctions à valeurs réelles et complexes, avec des applications aux fractales):
- la méthode de dichotomie et la méthode de la sécante ;
- la méthode de Newton (ou méthode des tangentes).
Pour chacune de ces méthodes, je devais aborder les aspects théoriques tels que la convergence, la stabilité et la complexité, tout en codant les algorithmes correspondants en Python.

Ensuite, je devais généraliser la méthode de Newton (Newton-Raphson) pour résoudre un système de n équations non linéaires à n inconnues x = (x1, ..., xn), ce qui revient à trouver un zéro d'une fonction F de Rn dans Rn, laquelle doit être différentiable. Dans cette approche, je devais utiliser la formule donnée suivante: JacF(xk)(xk+1 − xk) = −F(xk). En supposant que x0 soit assez proche du zéro pour que tout fonctionne.

Enfin, j'ai fait de ma propre initiative une digression sur l'algorithme de la fractale de Newton, tout en faisant quelques exemples appliqués de chaque point mentionné.
