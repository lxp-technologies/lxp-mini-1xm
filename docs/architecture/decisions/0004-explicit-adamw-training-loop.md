# ADR 0004 - Boucle explicite avec AdamW DJL

- Statut : accepté
- Date : 2026-08-30
- Portée : PR09

## Contexte

PR08 produit des logits et des gradients finis, mais cela ne prouve pas que les cibles, la loss, l'accumulation et l'optimizer sont raccordés correctement. Une abstraction d'entraînement de haut niveau réduirait le code, mais cacherait précisément les étapes que ce projet veut enseigner.

## Décision

Nous conservons une boucle Kotlin explicite pour :

- le forward et la cross-entropy next-token;
- le backward avec `GradientCollector`;
- la mise à l'échelle des gradients accumulés;
- le calcul et le clipping de la norme L2 globale;
- l'appel de l'optimizer et la remise à zéro;
- les compteurs d'updates, micro-batches et tokens.

Nous utilisons l'implémentation AdamW de DJL et un `Tracker` local pour le warmup linéaire suivi d'un cosine decay. La loss DJL reçoit les logits aplatis `[B*T,V]` et les cibles sparse `[B*T]`.

## Pourquoi

Cette séparation expose l'algorithme sans réécrire les opérations délicates déjà prises en charge par DJL/PyTorch. Elle permet aussi de tester isolément une cross-entropy connue, les bornes du scheduler, l'accumulation et l'overfit.

Une update ne doit avoir lieu qu'après le nombre configuré de micro-batches. Avant AdamW, tous les gradients partagent le même facteur de clipping global. Le learning rate imprimé est celui de l'update en cours, pas celui de la micro-batch.

## Conséquences

- Le code est plus long qu'un `EasyTrain.fit`, mais chaque transition reste observable.
- Les états AdamW sont internes à DJL; PR10 devra déterminer comment les sauvegarder ou déclarer précisément la limite.
- Le trainer suppose que le modèle a déjà été initialisé et que ses tenseurs résident sur le même device que les batches.
- `finishAccumulation()` doit être appelé lorsqu'un flux se termine avec un groupe partiel.
- Le nombre total d'updates est connu à la création du scheduler.
- PR09 reste sur FP32 et CPU dans son expérience de référence.

## Alternatives écartées

- **Trainer DJL de haut niveau :** moins de code, mais ordre des opérations moins visible pour ce laboratoire.
- **Réimplémenter AdamW :** valeur pédagogique faible comparée au risque d'erreur dans les moments et corrections de biais.
- **Clipping par composante :** ne conserve pas la direction du gradient global.
- **Softmax avant la loss :** calcul redondant et stabilité numérique inférieure.
- **Lancer directement un long run 17 M :** coûteux et incapable d'isoler rapidement une erreur de plomberie.

## Preuves exigées

- cross-entropy uniforme égale à `ln(V)`;
- gradient de logits fini pour un cas connu;
- aucune modification des poids avant la fin de l'accumulation;
- clipping global mesuré avant et après;
- scheduler exact au début, à la fin du warmup et à la dernière update;
- chute nette de loss sur un unique lot répété;
- libération du `NDManager` dans la commande exécutable.
