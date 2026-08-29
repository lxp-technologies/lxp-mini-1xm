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
./gradlew.bat test
./gradlew.bat run --args="model info --config configs/mini-17m.yaml"
```

Sous Linux ou macOS :

```bash
./gradlew test
./gradlew run --args="model info --config configs/mini-17m.yaml"
```

Le premier lancement peut télécharger Gradle, une toolchain JDK 25 et les dépendances JVM. Aucun backend PyTorch n'est téléchargé dans PR01.

## Expérience à essayer

1. Exécute `model info` avec `mini-11m`, `mini-14m`, `mini-17m`, puis `mini-22m`.
2. Copie localement un preset et passe `tieEmbeddings` à `false`.
3. Observe que `mini-17m` passe de `17 308 032` à `20 453 760` paramètres.
4. Essaie `dModel: 385` et lis l'erreur avant de remettre le fichier valide.

## Résultats attendus

Pour `mini-17m`, `headDim` vaut 64 et le total vaut exactement `17 308 032`. Une clé YAML inconnue, une dimension négative ou une largeur non divisible par les têtes produit une erreur et un code de sortie non nul.

## Questions de compréhension

1. Pourquoi `dModel` doit-il être divisible par `numHeads`?
2. Pourquoi `headDim` doit-il être pair pour notre futur RoPE?
3. Pourquoi augmenter `vocabSize` augmente-t-il les paramètres?
4. Pourquoi `contextLength` n'apparaît-il pas dans la formule actuelle?
5. Combien de paramètres économise le weight tying de `mini-17m`?

## Ce que la prochaine PR ajoutera

PR02 construira le byte tokenizer UTF-8. Nous verrons pour la première fois comment du texte devient des IDs et pourquoi 256 tokens suffisent à représenter tous les bytes possibles.
