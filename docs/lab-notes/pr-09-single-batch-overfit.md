# Laboratoire PR09 - Mémoriser un seul batch

Date : 2026-08-30<br>
Branche : `feature/pr09-loss-training-loop`

## Objectifs

Ce laboratoire permet de :

- relier les targets next-token aux logits avec la cross-entropy;
- observer backward, accumulation, clipping et AdamW;
- suivre warmup puis cosine decay;
- prouver que toute la chaîne apprend avant un entraînement coûteux;
- distinguer un sanity check de mémorisation d'une évaluation de généralisation.

## Exécuter PR09

Depuis la racine du dépôt sous PowerShell :

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.training.*"
.\gradlew.bat run --args="train overfit-batch --config configs/lab-pr09-tiny.yaml --updates 80 --report-every 10"
```

Sous Linux ou macOS, remplacer `.\gradlew.bat` par `./gradlew`.

Le preset utilise `V=259`, `C=16`, un bloc, deux têtes, une séquence de 8 tokens et un batch de 2. Il respecte le vocabulaire byte-level minimal, mais les IDs synthétiques n'exigent aucun corpus ni tokenizer.

## Sortie observée

```text
Batch shape:      (2, 8) = [B, T]
Optimizer:        AdamW
Accumulation:     1 micro-batch(es)/update
update=   1 loss=5.562087 lr=0.006000 grad=3.456999->1.000000 clipped=true  tokens=16
update=  10 loss=1.509770 lr=0.029806 grad=1.439417->1.000000 clipped=true  tokens=160
update=  20 loss=0.043364 lr=0.027685 grad=0.156599->0.156599 clipped=false tokens=320
update=  80 loss=0.000165 lr=0.003000 grad=0.000752->0.000752 clipped=false tokens=1280
Initial loss:     5.562087
Final loss:       0.000165
Reduction factor: 33803.964844x
Loss decreased:   true
Manager closed:   true
```

Les valeurs peuvent varier avec une autre version du moteur ou une autre machine, mais la loss doit rester finie et chuter fortement. À l'update 1, le clipping ramène bien `3.456999` à `1.0`. À l'update 80, le learning rate atteint le minimum configuré.

## Expérience 1 - Observer plus finement le warmup

```powershell
.\gradlew.bat run --args="train overfit-batch --config configs/lab-pr09-tiny.yaml --updates 20 --report-every 1"
```

Les cinq premiers learning rates doivent progresser linéairement vers `0.03`. La première update suivant le warmup reste au maximum, puis la décroissance commence. `--updates` doit être strictement supérieur à `warmupSteps`.

## Expérience 2 - Tester l'accumulation

Créer une copie locale du preset, puis modifier :

```powershell
New-Item -ItemType Directory -Force build/labs/pr09
Copy-Item configs/lab-pr09-tiny.yaml build/labs/pr09/accumulation.yaml
```

```yaml
training:
  batchSize: 1
  gradientAccumulationSteps: 2
```

Exécuter :

```powershell
.\gradlew.bat run --args="train overfit-batch --config build/labs/pr09/accumulation.yaml --updates 40 --report-every 10"
```

Chaque update consomme alors deux micro-batches. Le batch effectif reste deux séquences et `tokens` augmente de `1 * 8 * 2 = 16` par update. Les poids ne doivent changer qu'après la deuxième micro-batch; ce comportement est aussi protégé par test.

## Expérience 3 - Rendre le clipping visible

Dans une autre copie, régler temporairement :

```powershell
Copy-Item configs/lab-pr09-tiny.yaml build/labs/pr09/clipping.yaml
```

```yaml
training:
  gradientClipNorm: 0.1
```

Puis relancer cette copie avec `--report-every 1` :

```powershell
.\gradlew.bat run --args="train overfit-batch --config build/labs/pr09/clipping.yaml --updates 40 --report-every 1"
```

Lorsque `clipped=true`, la norme après la flèche doit valoir `0.100000`. La convergence peut devenir plus lente, car les premières updates sont davantage réduites.

## Exécuter tous les tests

```powershell
.\gradlew.bat test
```

Les 117 tests actuels couvrent notamment loss connue, gradients finis, formes invalides, scheduler, accumulation complète et partielle, clipping, overfit direct et overfit par la CLI.

## Questions et réponses

### Pourquoi la loss initiale est-elle proche de `ln(259)`?

Un modèle non entraîné répartit approximativement ses scores sur les 259 tokens. Une distribution uniforme attribue une probabilité `1/259` à la cible, donc une cross-entropy `-ln(1/259) = ln(259)`, environ `5.56`.

### Pourquoi répéter toujours le même batch?

Le but n'est pas de généraliser. Nous voulons rendre le problème volontairement facile. Si un petit modèle ne peut pas mémoriser 16 cibles répétées, un long entraînement masquerait probablement une erreur de targets, gradients, optimizer ou learning rate.

### Une loss presque nulle signifie-t-elle que le modèle est bon?

Non. Elle prouve seulement que la chaîne d'entraînement peut ajuster les poids à ce lot. Le modèle a surappris exactement ce que nous lui avons montré. Une validation sur des données séparées sera nécessaire en PR12.

### Pourquoi diviser la loss pendant l'accumulation?

Sans division, doubler `gradientAccumulationSteps` doublerait approximativement la norme du gradient et changerait l'update. La division produit la moyenne des micro-batches, comme un batch plus grand. Un dernier groupe partiel reçoit une correction explicite.

### Pourquoi utiliser AdamW de DJL plutôt que tout coder?

Le projet veut rendre la boucle visible, pas réimplémenter chaque primitive numérique. DJL gère les moments et leurs détails; notre code contrôle quand l'optimizer agit, quel learning rate il reçoit et quels gradients lui sont fournis.

### Puis-je déjà entraîner `mini-17m` sur mon dataset?

Pas encore avec une commande supportée de bout en bout. PR09 n'intègre que le test d'overfit synthétique. Les checkpoints arrivent en PR10 et la pipeline de vrai run avec validation en PR12. Attendre ces garde-fous évite de perdre des heures sur un run impossible à reprendre ou à évaluer.

## Prochaine étape

PR10 sauvegardera les poids et étudiera la restauration des états AdamW, du scheduler et des compteurs. La preuve attendue sera un round-trip dont les logits restent identiques.
