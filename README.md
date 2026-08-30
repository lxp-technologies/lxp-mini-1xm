# lxp-mini-1xm

Petit modèle de langage decoder-only construit progressivement from scratch en Kotlin/JVM. Chaque PR introduit un concept exécutable, testé et documenté.

## PR02 - Explorer le byte tokenizer

Affiche les bytes UTF-8 et les IDs produits pour un texte contenant un accent et un emoji :

```powershell
.\gradlew.bat run --args="tokenizer byte inspect --text-file docs/lab-notes/samples/pr02-unicode.txt --add-bos --add-eos"
```

Crée ensuite l'artefact versionné du tokenizer :

```powershell
.\gradlew.bat run --args="tokenizer byte create --output build/labs/pr02/tokenizer.json"
Get-Content build/labs/pr02/tokenizer.json
```

Sous Linux ou macOS, remplace `.\gradlew.bat` par `./gradlew` et utilise `cat` pour afficher le JSON. `--text-file` préserve les caractères Unicode indépendamment de l'encodage du terminal Windows. Le chapitre [Comprendre la tokenization](docs/02-tokenization.md) et la [note de laboratoire PR02](docs/lab-notes/pr-02-byte-tokenizer.md) expliquent chaque nombre affiché.

## PR01 - Exécuter les fondations

PR01 valide les configurations YAML et calcule le nombre théorique de paramètres. Elle ne contient pas encore de tokenizer ni de réseau neuronal.

### Prérequis

- une JVM installée pour démarrer Gradle;
- une connexion Internet au premier lancement pour télécharger Gradle, la toolchain JDK 25 et les dépendances JVM.

Il n'est pas nécessaire d'installer Gradle globalement : les commandes utilisent le Gradle Wrapper versionné dans le dépôt.

### Windows PowerShell

Depuis la racine du projet :

```powershell
.\gradlew.bat test
.\gradlew.bat run --args="model info --config configs/mini-17m.yaml"
```

### Linux et macOS

Depuis la racine du projet :

```bash
./gradlew test
./gradlew run --args="model info --config configs/mini-17m.yaml"
```

### Résultat attendu

Les tests doivent se terminer par `BUILD SUCCESSFUL`. La commande `model info` doit notamment afficher :

```text
Vocabulary:           8,192
Embedding size:       384
Layers:               8
Attention heads:      6
Head dimension:       64
FFN dimension:        1,024

Total parameters:     17,308,032
```

### Essayer les autres presets

Remplace le chemin de configuration par l'un des presets suivants :

```powershell
.\gradlew.bat run --args="model info --config configs/mini-11m.yaml"
.\gradlew.bat run --args="model info --config configs/mini-14m.yaml"
.\gradlew.bat run --args="model info --config configs/mini-22m.yaml"
```

Sous Linux ou macOS, remplace `.\gradlew.bat` par `./gradlew`.

### Provoquer une erreur pédagogique

Dans une copie locale de `configs/mini-17m.yaml`, remplace temporairement `dModel: 384` par `dModel: 385`, puis exécute `model info` avec cette copie. La commande doit refuser la configuration parce que `dModel` n'est pas divisible par `numHeads`.

## Documentation

- [Parcours d'apprentissage](docs/README.md)
- [Note de laboratoire PR01](docs/lab-notes/pr-01-project-foundation.md)
- [Calcul des paramètres](docs/architecture/parameter-counting.md)
- [Plan directeur](docs/plan-directeur.md)
