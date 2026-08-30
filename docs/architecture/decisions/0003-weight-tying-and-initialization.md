# ADR 0003 - Weight tying réel et initialisation explicite

- Statut : accepté
- Date : 2026-08-30
- Portée : PR08

## Contexte

Le modèle possède une table d'embedding `[V,C]` pour convertir les IDs en états. Le LM head doit convertir les états finaux `[B,T,C]` en logits `[B,T,V]`. Une implémentation naïve ajouterait une matrice `[C,V]` indépendante, alors que le plan prévoit le partage des poids par défaut.

Le partage doit être réel. Deux tableaux initialisés avec les mêmes valeurs ne sont pas liés : une mise à jour de l'un ne modifierait pas l'autre.

## Décision

En mode `tieEmbeddings=true`, `LanguageModelHead` :

- référence exactement le même objet `Parameter` que `TokenEmbedding`;
- obtient exactement le même objet `NDArray` via le `ParameterStore`;
- utilise sa transposée pendant `hidden × embeddingWeightᵀ`;
- n'enregistre aucun paramètre direct supplémentaire.

En mode non lié, le head enregistre sa propre matrice `[C,V]`.

L'initialisation de référence demeure :

- embeddings et head non lié : normale d'écart-type `0,02`;
- projections attention et SwiGLU : Xavier;
- échelles RMSNorm : 1;
- aucun biais, aucune table positionnelle apprise.

## Pourquoi

L'embedding apprend une représentation pour chaque token. Le head apprend quelles représentations finales favorisent chaque token en sortie. Partager les deux espaces réduit les paramètres et impose une cohérence utile entre lecture et prédiction du vocabulaire.

Le partage économise `V×C` poids. Pour `V=8192`, `C=384`, cela représente `3 145 728` paramètres, environ `12,6 MB` de poids FP32 avant de compter gradients et états optimizer.

Les initialisations actuelles ont déjà été validées composant par composant et produisent un forward fini sur le modèle complet. PR09 apportera la preuve plus forte du single-batch overfit; toute modification future devra conserver les tests numériques et documenter son effet.

## Preuves exigées

- identité Kotlin du `Parameter` avec `===`;
- identité du `NDArray` initialisé avec `===`;
- aucun paramètre direct sur le head lié;
- un paramètre `[C,V]` sur le head non lié;
- différence réelle et théorique exactement égale à `V×C`;
- gradients finis sur le poids partagé depuis les logits.

## Conséquences

- Le head lié utilise la forme transposée au calcul, sans copier les données.
- Une mise à jour optimizer du poids partagé affectera simultanément embedding et sortie.
- Le paramètre ne doit être enregistré qu'une fois, sinon un optimizer pourrait le mettre à jour deux fois.
- Le mode non lié reste disponible comme expérience contrôlée et augmente fortement la mémoire.
- Dropout non nul est refusé tant que sa sémantique n'est pas implémentée et testée.

## Alternatives écartées

- **Copier les valeurs de l'embedding :** égalité initiale seulement, pas un partage durable.
- **Enregistrer le même paramètre dans deux enfants :** risque de double comptage et de double mise à jour.
- **Toujours imposer le partage :** empêcherait l'expérience demandée et masquerait son économie réelle.
- **Ajouter softmax au modèle :** la cross-entropy attend des logits bruts et sera construite en PR09.
