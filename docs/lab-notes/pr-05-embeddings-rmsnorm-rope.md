# PR05 - Embeddings, RMSNorm et RoPE

## Ce que nous avons construit

PR05 ajoute DJL `0.36.0`, le moteur PyTorch, une table d'embedding explicite, RMSNorm explicite, RoPE explicite, les gradients et une gestion déterministe des ressources natives.

## Prérequis du premier lancement

La première exécution peut télécharger les dépendances Maven, le JNI DJL et les bibliothèques natives PyTorch adaptées à la machine. Une connexion Internet est donc requise une fois, sauf si ces artefacts sont déjà en cache ou configurés pour un usage hors ligne.

Sous Windows PowerShell :

```powershell
.\gradlew.bat test --tests '*DjlEngineTest'
```

Le test doit charger `PyTorch` sur `cpu()` et terminer par `BUILD SUCCESSFUL`.

## Expérience 1 - Exécuter tous les composants

```powershell
.\gradlew.bat run --args="model components --vocab-size 32 --d-model 8 --num-heads 2 --batch-size 2 --sequence-length 4 --context-length 16 --rope-theta 10000 --seed 42"
```

Résultats observés le 2026-08-29 :

```text
DJL engine:          PyTorch 2.7.1
Token IDs shape:     (2, 4) = [B, T]
Embedding shape:     (2, 4, 8) = [B, T, C]
Heads shape:         (2, 2, 4, 4) = [B, H, T, D]
RoPE cache shape:    (16, 2) = [context, D/2]
RoPE parameters:     0
Embedding weights:   256
RMSNorm weights:     8
Manager closed:      true
```

La table contient `32 × 8 = 256` poids. RMSNorm ajoute 8 poids. RoPE n'en ajoute aucun.

## Expérience 2 - Vérifier une rotation à la main

La commande utilise `[1,0,1,0]` aux positions 0 et 1 :

```text
position 0 après = [1.000000, 0.000000, 1.000000, 0.000000]
position 1 après = [0.540302, 0.841471, 0.999950, 0.010000]
```

Pour `D=4` et `theta=10000` :

```text
paire 0 : angle = 1 × 10000^0     = 1
           cos(1) = 0.540302, sin(1) = 0.841471
paire 1 : angle = 1 × 10000^(-1)  = 0.01
           cos(0.01) = 0.999950, sin(0.01) = 0.010000
```

La position 0 reste identique parce que ses deux angles valent zéro.

## Expérience 3 - Exécuter les preuves numériques ciblées

```powershell
.\gradlew.bat test --tests '*TokenEmbeddingTest'
.\gradlew.bat test --tests '*RmsNormTest'
.\gradlew.bat test --tests '*RotaryPositionEmbeddingTest'
```

Ces tests vérifient respectivement le lookup et l'accumulation des gradients, la formule RMS manuelle, les rotations connues, les offsets de position, les gradients et la fermeture du cache manager.

## Expérience 4 - Provoquer une erreur de forme

```powershell
.\gradlew.bat run --args="model components --d-model 10 --num-heads 2"
```

Ici `D=10/2=5`, qui est impair. L'application retourne le code 2 et explique que `head dimension must be even for RoPE`. Gradle présente alors normalement la tâche `run` comme `FAILED` et retourne 1 au terminal, puisque son processus enfant a refusé la configuration; ce résultat est l'expérience attendue, pas une erreur de compilation.

## Expérience 5 - Comparer le nombre de poids du preset

Pour `V=8192` et `C=384` :

```text
embedding = 8192 × 384 = 3 145 728
un RMSNorm = 384
RoPE = 0
```

La commande de laboratoire utilise volontairement de petites dimensions pour rester lisible. Le compteur complet du preset continue d'être vérifié avec :

```powershell
.\gradlew.bat run --args="model info --config configs/mini-17m.yaml"
```

## Questions de compréhension

1. Pourquoi l'embedding produit-il une dimension supplémentaire `C`?
2. Pourquoi un ID répété deux fois accumule-t-il deux contributions de gradient?
3. Quelle différence essentielle sépare RMSNorm de LayerNorm?
4. Pourquoi RoPE ne change-t-il rien à la position 0?
5. Pourquoi RoPE exige-t-il un `headDim` pair?
6. Pourquoi les caches RoPE ne comptent-ils pas comme paramètres?
7. Pourquoi devons-nous fermer les `NDManager` explicitement?

## Réponses et explications

### 1. Pourquoi ajouter `C`?

Chaque scalaire ID devient une ligne de `C` nombres appris. Les axes batch et temps restent inchangés, donc `[B,T]` devient `[B,T,C]`.

### 2. Pourquoi les gradients s'additionnent-ils?

La même ligne de poids a contribué à deux positions du forward. La dérivée totale d'un poids est la somme de toutes les voies du graphe qui utilisent ce poids.

### 3. RMSNorm ou LayerNorm?

RMSNorm divise par la racine de la moyenne des carrés et applique `gamma`. LayerNorm soustrait aussi la moyenne et possède généralement un biais appris. Les deux opérations ne produisent donc pas la même sortie.

### 4. Pourquoi la position 0 est-elle l'identité?

Chaque angle contient un facteur `position`. À zéro, tous les angles valent zéro; la matrice de rotation devient l'identité.

### 5. Pourquoi `D` doit-il être pair?

La rotation travaille sur des plans 2D. Chaque composante doit avoir une partenaire; une dimension impaire laisserait une composante sans règle.

### 6. Pourquoi les caches ne sont-ils pas des poids?

Ils sont recalculables exactement à partir de trois constantes : position, `theta` et `D`. Aucun optimizer ne les modifie et aucun gradient n'est requis.

### 7. Pourquoi fermer les managers?

Les arrays possèdent de la mémoire native que le garbage collector JVM ne libère pas au rythme précis des batches. Les scopes explicites évitent qu'activations et gradients s'accumulent jusqu'à épuiser la RAM ou la VRAM.

## Validation complète

```powershell
.\gradlew.bat check
```

Sous Linux ou macOS, remplace `.\gradlew.bat` par `./gradlew`.

## Ce que la prochaine PR ajoutera

PR06 construira les projections `Q`, `K`, `V`, appliquera réellement RoPE à `Q/K`, créera le masque causal et démontrera qu'un token futur ne peut modifier aucune sortie passée.
