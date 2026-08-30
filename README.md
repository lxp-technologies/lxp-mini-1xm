# lxp-mini-1xm

Petit modèle de langage decoder-only construit progressivement from scratch en Kotlin/JVM. Chaque PR introduit un concept exécutable, testé et documenté.

## PR10 - Sauvegarder et vérifier un checkpoint

Interrompt un tiny run, recrée le modèle, prouve des logits identiques et poursuit avec les limites de reprise affichées :

```powershell
$runDir = "build/labs/pr10/demo-$(Get-Date -Format yyyyMMdd-HHmmss)"
.\gradlew.bat run --args="train checkpoint-demo --config configs/lab-pr09-tiny.yaml --run-dir $runDir --before-updates 10 --after-updates 5"
.\gradlew.bat run --args="train checkpoint-verify --run-dir $runDir"
```

Chaque exécution doit utiliser un nouveau dossier : PR10 refuse volontairement d'écraser un run existant. Les poids, compteurs et scheduler sont restaurés; les moments AdamW et l'état aléatoire ne le sont pas, donc `exactTrainingResume=false`. Consulte le [chapitre PR10](docs/10-checkpoints-and-reproducible-runs.md) et la [note de laboratoire](docs/lab-notes/pr-10-checkpoint-round-trip.md).

## PR09 - Observer un modèle qui apprend

Répète un lot synthétique sur un tiny model et affiche cross-entropy, learning rate, norme globale des gradients, clipping et tokens vus :

```powershell
.\gradlew.bat run --args="train overfit-batch --config configs/lab-pr09-tiny.yaml --updates 80 --report-every 10"
```

La loss de référence chute d'environ `5.56` à moins de `0.05`. Ce sanity check valide backward, accumulation, AdamW et warmup/cosine avant tout long entraînement. Consulte le [chapitre PR09](docs/09-loss-and-training-loop.md) et la [note de laboratoire](docs/lab-notes/pr-09-single-batch-overfit.md).

## PR08 - Exécuter le modèle decoder-only complet

Instancie réellement les 17,3 M paramètres et produit des logits `[B,T,V]` :

```powershell
.\gradlew.bat run --args="model forward --config configs/mini-17m.yaml --batch-size 1 --sequence-length 4 --seed 42"
```

Ajoute `--untie-embeddings` pour mesurer la seconde matrice du LM head. La commande vérifie le partage par identité d'objet et compare le compte réel au [compteur théorique](docs/architecture/parameter-counting.md). Consulte la [note de laboratoire PR08](docs/lab-notes/pr-08-decoder-language-model.md).

## PR07 - Observer un bloc Transformer complet

Exécute un bloc pre-norm isolé avec attention causale, SwiGLU et deux connexions résiduelles :

```powershell
.\gradlew.bat run --args="model block --d-model 8 --num-heads 2 --ffn-dim 16 --sequence-length 4 --context-length 16 --seed 42"
```

La commande affiche toutes les formes, les `656` paramètres du bloc miniature, la causalité conservée et compare les gradients avec et sans résidus. Consulte la [note de laboratoire PR07](docs/lab-notes/pr-07-transformer-block.md).

## PR06 - Observer l'attention causale

Exécute une attention multi-tête sur quatre tokens et imprime la matrice de probabilités d'une tête :

```powershell
.\gradlew.bat run --args="model attention --d-model 8 --num-heads 2 --sequence-length 4 --context-length 16 --seed 42"
```

La commande vérifie également que chaque ligne somme à 1, que la probabilité du futur vaut 0 et que modifier le dernier token ne change aucune sortie passée. Consulte la [note de laboratoire PR06](docs/lab-notes/pr-06-causal-self-attention.md) pour refaire les expériences.

## PR05 - Observer embeddings, RMSNorm et RoPE

PR05 introduit DJL/PyTorch et les premiers tenseurs entraînables :

```powershell
.\gradlew.bat run --args="model components --vocab-size 32 --d-model 8 --num-heads 2 --batch-size 2 --sequence-length 4 --context-length 16 --rope-theta 10000 --seed 42"
```

La commande affiche les formes `[B,T,C]` et `[B,H,T,D]`, une rotation RoPE calculable, les normes de gradients et la fermeture du `NDManager`. Consulte la [note de laboratoire PR05](docs/lab-notes/pr-05-embeddings-rmsnorm-rope.md) pour les expériences.

## PR04 - Inspecter les séquences d'entraînement

Observe d'abord le décalage next-token sans tokenizer :

```powershell
.\gradlew.bat run --args="dataset window --tokens 10,20,30,40,50 --context-length 4"
```

Puis entraîne le tokenizer de laboratoire et parcours réellement le corpus par streaming :

```powershell
.\gradlew.bat run --args="tokenizer bpe train --input docs/lab-notes/samples/pr03-corpus.txt --vocab-size 272 --output build/labs/pr04/tokenizer.json"
.\gradlew.bat run --args="dataset inspect --corpus docs/lab-notes/samples/pr03-corpus.txt --tokenizer build/labs/pr04/tokenizer.json --context-length 8 --batch-size 3 --validation-fraction 0.2 --split train --shuffle-buffer 4 --seed 42 --show-batches 2"
```

La [note de laboratoire PR04](docs/lab-notes/pr-04-dataset-and-sequences.md) explique les formes `[B,T]`, le split sans fuite, les fragments finaux et la mémoire bornée.

## PR03 - Entraîner un byte-level BPE

Entraîne un petit tokenizer déterministe sur le corpus de laboratoire, puis inspecte les pièces apprises :

```powershell
.\gradlew.bat run --args="tokenizer bpe train --input docs/lab-notes/samples/pr03-corpus.txt --vocab-size 272 --output build/labs/pr03/tokenizer.json"
.\gradlew.bat run --args="tokenizer bpe inspect --tokenizer build/labs/pr03/tokenizer.json --text 'Bonjour bonjour' --show-merges 13 --show-vocabulary 13"
```

Le vocabulaire minimal vaut 259, car il inclut `PAD`, `BOS`, `EOS` et les 256 bytes. La [note de laboratoire PR03](docs/lab-notes/pr-03-byte-level-bpe.md) explique comment comparer plusieurs tailles et interpréter le ratio bytes/token.

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
