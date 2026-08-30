# Laboratoire PR07 - Assembler et inspecter un bloc Transformer

Date : 2026-08-30<br>
Branche : `feature/pr07-transformer-block`

## Objectifs

Ce laboratoire permet de :

- suivre toutes les formes d'un bloc pre-norm;
- vérifier le calcul `3CF` de SwiGLU et le total du bloc;
- constater que les sorties demeurent finies;
- répéter la preuve anti-fuite après l'assemblage;
- comparer le gradient d'entrée avec et sans les deux résidus.

## Exécuter PR07

Sous Windows PowerShell, depuis la racine :

```powershell
.\gradlew.bat run --args="model block --d-model 8 --num-heads 2 --ffn-dim 16 --sequence-length 4 --context-length 16 --seed 42"
```

Sous Linux ou macOS, remplace `.\gradlew.bat` par `./gradlew`.

## Sortie observée

```text
Input shape:               (1, 4, 8) = [B, T, C]
Attention norm shape:      (1, 4, 8)
Attention output shape:    (1, 4, 8)
First residual shape:      (1, 4, 8)
Feed-forward norm shape:   (1, 4, 8)
SwiGLU hidden shape:       (1, 4, 16) = [B, T, F]
Feed-forward output shape: (1, 4, 8)
Block output shape:        (1, 4, 8) = [B, T, C]
Attention parameters:      256
SwiGLU parameters:         384
RMSNorm parameters:        16
Block parameters:          656
Output finite:             true
Past output max delta:     0.000000
Gradient paths differ:     true
Manager closed:            true
```

Les normes exactes des gradients peuvent varier selon le moteur ou l'initialisation. Le laboratoire vérifie qu'elles sont finies et différentes, pas qu'une valeur particulière est universelle.

## Expérience 1 - Calculer les paramètres à la main

Avec `C=8` et `F=16` :

```text
attention = 4 × 8²       = 256
SwiGLU    = 3 × 8 × 16   = 384
RMSNorm   = 2 × 8        = 16
total                    = 656
```

Essaie une dimension cachée deux fois plus grande :

```powershell
.\gradlew.bat run --args="model block --d-model 8 --num-heads 2 --ffn-dim 32 --sequence-length 4 --context-length 16 --seed 42"
```

L'attention et RMSNorm restent inchangés. SwiGLU passe à `3×8×32=768`, donc le bloc à `1040` paramètres.

## Expérience 2 - Comparer les gradients résiduels

La commande effectue deux backward avec les mêmes poids et la même entrée :

1. forward normal avec `x + attention` puis `a + SwiGLU`;
2. forward d'inspection où les deux additions résiduelles sont retirées;
3. comparaison des normes du gradient par rapport à l'entrée.

Ne conclus pas « résidu = gradient toujours plus grand ». Le résultat important est l'existence d'un chemin identité et le changement mesuré du gradient. La direction des contributions détermine leur norme combinée.

Le test structurel va plus loin : il force les projections de sortie des deux sous-couches à zéro. Avec résidus, la sortie devient exactement l'entrée; sans résidus, elle devient exactement zéro.

## Expérience 3 - Vérifier le preset 17 M

Nous pouvons exécuter un seul bloc avec les dimensions finales sans construire le modèle entier :

```powershell
.\gradlew.bat run --args="model block --d-model 384 --num-heads 6 --ffn-dim 1024 --sequence-length 4 --context-length 256 --seed 42"
```

Le compte attendu est :

```text
attention = 589 824
SwiGLU    = 1 179 648
RMSNorm   = 768
bloc      = 1 770 240
```

Cette commande n'entraîne rien et n'empile qu'un bloc. Elle valide les dimensions et le compte réel avant PR08.

## Expérience 4 - Provoquer une erreur de forme

```powershell
.\gradlew.bat run --args="model block --d-model 12 --num-heads 4 --ffn-dim 16"
```

Ici `D=12/4=3`, impair, donc RoPE ne peut pas former ses paires. L'application retourne 2 et Gradle signale l'échec attendu du programme lancé.

## Exécuter les tests

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.model.SwiGluFeedForwardTest"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.model.TransformerBlockTest"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.cli.ModelBlockCommandTest"
.\gradlew.bat check
```

Les tests protègent :

- une valeur SwiGLU calculée manuellement avec des matrices identité;
- les formes `[B,T,C]` et `[B,T,F]`;
- les trois gradients SwiGLU et tous les gradients d'entrée finis;
- les neuf paramètres appris du bloc et leur nombre total de poids;
- le comportement identité des résidus lorsque les sous-couches valent zéro;
- la causalité du bloc complet;
- la fermeture du cache RoPE sans fermer le manager parent.

## Questions et réponses

### Pourquoi le FFN ne mélange-t-il pas les tokens?

Les mêmes matrices sont appliquées séparément à chaque position `[b,t,:]`. Elles mélangent les composantes de `C` et `F`, mais aucune opération n'utilise un autre indice `t`.

### Pourquoi `F` est-il plus grand que `C`?

L'expansion donne à chaque position un espace intermédiaire plus riche pour construire des combinaisons non linéaires. Elle augmente toutefois directement les paramètres et le calcul via `3CF`.

### Pourquoi deux RMSNorm plutôt qu'une seule?

L'attention et SwiGLU reçoivent chacune une entrée normalisée différente. Après le premier résidu, l'état a changé; la seconde normalisation prépare cet état pour le feed-forward.

### Les résidus ajoutent-ils des paramètres?

Non. Une connexion résiduelle est une addition de tenseurs de même forme. Elle ajoute du calcul et conserve des activations pour le backward, mais aucun poids appris.

### Pourquoi tester encore la causalité?

Une primitive correcte peut être mal assemblée. Répéter le test anti-fuite au niveau supérieur protège le contrat utilisateur du bloc, pas seulement l'implémentation interne de l'attention.

## Prochaine étape

PR08 reliera les embeddings, une pile de blocs, le RMSNorm final et le LM head avec partage réel des poids pour produire `[B,T,V]` logits.
