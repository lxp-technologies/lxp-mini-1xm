# PR01 - Fondations et configuration du modèle

## Ce que nous avons construit

Un projet Kotlin/JVM reproductible, quatre presets YAML stricts, leurs règles de validation, un compteur théorique de paramètres, une CLI, des tests et une CI. Aucun réseau neuronal n'existe encore.

## Ce qu'il faut comprendre

La configuration décrit une famille de modèles. Changer la taille doit modifier le YAML, jamais les formules du code. Une validation fail-fast empêche de consacrer du temps ou du GPU à une expérience impossible.

## Formules importantes

```text
headDim = dModel / numHeads
total = V×C + N×(4C² + 3CF + 2C) + C
```

Le LM head ajoute `V×C` seulement lorsque `tieEmbeddings=false`.

## Tensor shapes importantes

PR01 ne crée aucun tenseur. Elle prépare toutefois `C=dModel=384`, `H=numHeads=6` et donc `D=headDim=64`. Les premières formes réelles arriveront avec les embeddings en PR05.

## Comment l'exécuter

Sous Windows :

```powershell
.\gradlew.bat test
.\gradlew.bat run --args="model info --config configs/mini-17m.yaml"
```

Sous Linux ou macOS :

```bash
./gradlew test
./gradlew run --args="model info --config configs/mini-17m.yaml"
```

Le premier lancement peut télécharger Gradle, une toolchain JDK 25 et les dépendances JVM. Aucun backend PyTorch n'est téléchargé dans PR01.

## Expériences à exécuter

Les commandes PowerShell suivantes partent de la racine du projet. Les copies expérimentales sont placées sous `build/labs/pr01/`, un dossier ignoré par Git. Les presets dans `configs/` restent donc intacts.

### Expérience 1 - Vérifier PR01

```powershell
.\gradlew.bat clean test
.\gradlew.bat run --args="model info --config configs/mini-17m.yaml"
```

Résultat attendu : les tests se terminent par `BUILD SUCCESSFUL`. Le rapport indique `headDim = 64` et `Total parameters = 17,308,032`.

### Expérience 2 - Comparer les quatre presets

```powershell
.\gradlew.bat run --args="model info --config configs/mini-11m.yaml"
.\gradlew.bat run --args="model info --config configs/mini-14m.yaml"
.\gradlew.bat run --args="model info --config configs/mini-17m.yaml"
.\gradlew.bat run --args="model info --config configs/mini-22m.yaml"
```

| Preset     | Couches | `dModel` | Paramètres attendus |
|------------|--------:|---------:|--------------------:|
| `mini-11m` |       6 |      320 |          10 981 440 |
| `mini-14m` |       6 |      384 |          13 767 552 |
| `mini-17m` |       8 |      384 |          17 308 032 |
| `mini-22m` |      11 |      384 |          22 618 752 |

Compare surtout `mini-14m`, `mini-17m` et `mini-22m` : seule la profondeur change. Le nombre de paramètres augmente donc d'une quantité constante par bloc ajouté.

### Expérience 3 - Mesurer le weight tying

Crée une copie du preset, désactive le partage, puis inspecte-la :

```powershell
New-Item -ItemType Directory -Force build/labs/pr01
Copy-Item configs/mini-17m.yaml build/labs/pr01/mini-17m-untied.yaml
$untiedConfig = Get-Content -Raw build/labs/pr01/mini-17m-untied.yaml
$untiedConfig.Replace("tieEmbeddings: true", "tieEmbeddings: false") | Set-Content -Encoding utf8 build/labs/pr01/mini-17m-untied.yaml
.\gradlew.bat run --args="model info --config build/labs/pr01/mini-17m-untied.yaml"
```

Résultat attendu : `Output head = 3,145,728` et `Total parameters = 20,453,760`. La différence avec le preset partagé est exactement :

```text
vocabSize × dModel = 8 192 × 384 = 3 145 728
```

### Expérience 4 - Observer la validation fail-fast

Crée une autre copie, puis rends `dModel` incompatible avec les six têtes :

```powershell
New-Item -ItemType Directory -Force build/labs/pr01
Copy-Item configs/mini-17m.yaml build/labs/pr01/mini-17m-invalid.yaml
$invalidConfig = Get-Content -Raw build/labs/pr01/mini-17m-invalid.yaml
$invalidConfig.Replace("dModel: 384", "dModel: 385") | Set-Content -Encoding utf8 build/labs/pr01/mini-17m-invalid.yaml
.\gradlew.bat run --args="model info --config build/labs/pr01/mini-17m-invalid.yaml"
```

Résultat attendu : la commande affiche `model.dModel must be divisible by model.numHeads` et Gradle termine en échec parce que la CLI retourne volontairement un code non nul. Aucun entraînement ne peut commencer avec cette configuration.

### Nettoyer les fichiers expérimentaux

```powershell
Remove-Item -Recurse -Force build/labs/pr01
```

Sous Linux ou macOS, les deux transformations équivalentes sont :

```bash
mkdir -p build/labs/pr01
sed 's/tieEmbeddings: true/tieEmbeddings: false/' configs/mini-17m.yaml > build/labs/pr01/mini-17m-untied.yaml
./gradlew run --args="model info --config build/labs/pr01/mini-17m-untied.yaml"

sed 's/dModel: 384/dModel: 385/' configs/mini-17m.yaml > build/labs/pr01/mini-17m-invalid.yaml
./gradlew run --args="model info --config build/labs/pr01/mini-17m-invalid.yaml"
```

## Questions de compréhension

1. Pourquoi `dModel` doit-il être divisible par `numHeads`?
2. Pourquoi `headDim` doit-il être pair pour notre futur RoPE?
3. Pourquoi augmenter `vocabSize` augmente-t-il les paramètres?
4. Pourquoi `contextLength` n'apparaît-il pas dans la formule actuelle?
5. Combien de paramètres économise le weight tying de `mini-17m`?

## Réponses et explications

### 1. Pourquoi `dModel` doit-il être divisible par `numHeads`?

Chaque token possède un vecteur de `dModel` composantes que l'attention sépare également entre les têtes :

```text
headDim = dModel / numHeads
384 / 6 = 64
```

Chaque tête reçoit donc 64 dimensions. Sans divisibilité exacte, les dimensions ne pourraient pas être réparties uniformément puis fusionnées pour reconstruire un vecteur de taille `dModel`.

### 2. Pourquoi `headDim` doit-il être pair pour RoPE?

RoPE regroupe les composantes de Q et K par paires et applique une rotation 2D à chaque paire. Avec `headDim = 64`, une tête contient 32 paires. Une dimension impaire laisserait une composante sans partenaire; notre implémentation choisit donc de rejeter cette configuration explicitement.

### 3. Pourquoi augmenter `vocabSize` augmente-t-il les paramètres?

Chaque token du vocabulaire possède un vecteur appris de taille `dModel`. La table contient donc `vocabSize × dModel` poids. Ajouter un token ajoute exactement `dModel` poids à l'embedding, et en ajouterait encore `dModel` au LM head si ses poids n'étaient pas partagés.

### 4. Pourquoi `contextLength` n'apparaît-il pas dans la formule actuelle?

RoPE n'utilise pas de table de positions entraînable. Allonger le contexte ne crée donc aucun nouveau poids. Cela augmente toutefois fortement les activations, la mémoire et le calcul : les scores d'attention ont la forme `[B, H, T, T]`, donc leur coût croît approximativement comme `T²`.

Avec des embeddings positionnels appris, la réponse serait différente : une table `[contextLength, dModel]` ajouterait `contextLength × dModel` paramètres.

### 5. Combien de paramètres économise le weight tying de `mini-17m`?

Le LM head aurait autrement une seconde matrice `[dModel, vocabSize]` :

```text
8 192 × 384 = 3 145 728 paramètres
```

Le partage fait donc passer le total de `20 453 760` à `17 308 032`, soit une économie exacte de `3 145 728` paramètres, environ 3,15 millions.

## Ce que la prochaine PR ajoutera

PR02 construira le byte tokenizer UTF-8. Nous verrons pour la première fois comment du texte devient des IDs et pourquoi 256 tokens suffisent à représenter tous les bytes possibles.
