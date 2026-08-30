# Laboratoire PR10 - Interrompre et restaurer un tiny run

Date : 2026-08-30<br>
Branche : `feature/pr10-checkpoints-reproducible-runs`

## Objectifs

Ce laboratoire permet de :

- créer l'arborescence complète d'un run;
- interrompre après 10 updates et fermer le premier modèle;
- restaurer les poids dans un nouveau manager;
- prouver que les logits sont identiques;
- continuer les compteurs et le scheduler jusqu'à l'update 15;
- observer pourquoi cette continuation AdamW n'est pas exacte;
- provoquer et détecter deux corruptions.

## Exécuter PR10

Choisir un nom de run neuf :

```powershell
.\gradlew.bat run --args="train checkpoint-demo --config configs/lab-pr09-tiny.yaml --run-dir build/labs/pr10/demo-001 --before-updates 10 --after-updates 5"
```

La commande refuse un dossier non vide. Pour une deuxième exécution, utiliser par exemple `demo-002`; elle ne supprime jamais un run existant.

## Sortie observée

```text
Interrupted at update:     10
Resumed through update:    15
Tokens seen:               240
Logits exactly identical:  true
Maximum logit difference:  0.0
Optimizer counter restored: true
Scheduler restored:         true
AdamW moments restored:     false
Random state restored:      false
Exact training resume:      false
Managers closed:            true
```

Le SHA-256 imprimé dépend des poids produits par le moteur. L'invariant important est l'identité des logits après fermeture et recréation, pas une valeur de checksum copiée depuis une autre machine.

## Inspecter le run

```powershell
Get-ChildItem build/labs/pr10/demo-001 -Recurse
Get-Content build/labs/pr10/demo-001/run-metadata.json
Get-Content build/labs/pr10/demo-001/metrics.jsonl | Select-Object -First 2
Get-Content build/labs/pr10/demo-001/metrics.jsonl | Select-Object -Last 2
Get-Content build/labs/pr10/demo-001/checkpoints/latest.txt
Get-Content build/labs/pr10/demo-001/checkpoints/step-00000010/manifest.json
```

`metrics.jsonl` doit contenir 15 lignes. Les dix premières portent `phase=before-checkpoint`; les cinq dernières, `phase=after-resume`. `latest.txt` pointe finalement sur `step-00000015`.

## Vérifier sans entraîner

```powershell
.\gradlew.bat run --args="train checkpoint-verify --run-dir build/labs/pr10/demo-001"
```

Cette commande recharge le dernier checkpoint, valide configuration et poids, puis affiche :

```text
Optimizer updates:          15
Tokens seen:                240
Model initialized:          true
AdamW moments restored:     false
Exact training resume:      false
Manager closed:             true
```

## Expérience 1 - Corrompre un octet

Travailler sur une copie afin de conserver le run original :

```powershell
Copy-Item -Recurse build/labs/pr10/demo-001 build/labs/pr10/corrupted-model
$model = "build/labs/pr10/corrupted-model/checkpoints/step-00000015/model.params"
$bytes = [System.IO.File]::ReadAllBytes($model)
$bytes[$bytes.Length - 1] = $bytes[$bytes.Length - 1] -bxor 1
[System.IO.File]::WriteAllBytes($model, $bytes)
.\gradlew.bat run --args="train checkpoint-verify --run-dir build/labs/pr10/corrupted-model"
```

Résultat attendu : l'application retourne 2 et affiche `Model checksum mismatch`; Gradle rapporte donc normalement l'échec du processus lancé. Le chargeur refuse les poids avant de les injecter dans le modèle.

## Expérience 2 - Modifier la configuration copiée

```powershell
Copy-Item -Recurse build/labs/pr10/demo-001 build/labs/pr10/corrupted-config
Add-Content build/labs/pr10/corrupted-config/config.yaml "# modification locale"
.\gradlew.bat run --args="train checkpoint-verify --run-dir build/labs/pr10/corrupted-config"
```

Le YAML demeure valide, mais son SHA-256 change. Le manifeste refuse donc l'association avec `configuration checksum does not match this run`.

## Expérience 3 - Comparer les deux checkpoints

```powershell
Get-Content build/labs/pr10/demo-001/checkpoints/step-00000010/manifest.json
Get-Content build/labs/pr10/demo-001/checkpoints/step-00000015/manifest.json
```

Observer `optimizerUpdates`, `tokensSeen` et `modelSha256`. Les checksums de poids diffèrent parce que cinq updates supplémentaires ont été appliquées. Les drapeaux de reprise restent identiques et honnêtes dans les deux manifestes.

## Exécuter les tests

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.checkpoint.*"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.cli.CheckpointDemoCommandTest"
.\gradlew.bat check
```

Les tests couvrent round-trip, logits identiques, backward après chargement, SHA corrompu, mauvaise configuration, dossier de run non vide, JSONL et commandes de démonstration/vérification.

Après PR10, la suite complète contient 124 tests.

## Questions et réponses

### Pourquoi les logits peuvent-ils être exacts si la reprise ne l'est pas?

Les logits dépendent des poids, de l'entrée et du forward. La prochaine update dépend en plus de l'historique AdamW. Restaurer les poids suffit donc à l'inférence, mais pas à reproduire l'optimisation.

### Que sont les moments AdamW?

AdamW maintient pour chaque poids une moyenne mobile du gradient `m` et de son carré `v`. Après dix updates, ces tableaux contiennent l'historique récent. Les remettre à zéro modifie la prochaine update même si le gradient courant est identique.

### Pourquoi restaurer le compteur AdamW sans ses moments?

Cela conserve au moins la correction de biais au bon numéro et évite de prétendre revenir à l'update 1. La continuation reste approximative, ce que le manifeste et la CLI déclarent.

### Pourquoi copier la configuration dans le run?

Le fichier global peut changer après l'expérience. La copie locale fixe les hyperparamètres réellement associés aux poids; son checksum empêche de charger accidentellement une autre architecture ou un autre schedule.

### Le SHA-256 sécurise-t-il le modèle contre un attaquant?

Non. Un attaquant pouvant remplacer les poids et le manifeste peut recalculer le hash. Le checksum protège ici l'intégrité accidentelle et l'association des fichiers. Une provenance authentifiée demanderait une signature et une gestion de clés.

### Puis-je utiliser le checkpoint pour PR11?

Oui. PR11 recharge ces poids avec `generate`; il faut lui fournir le byte tokenizer séparé. Les moments AdamW ne sont pas requis pour l'inférence.

## Prochaine étape

Exécuter maintenant le [laboratoire PR11](pr-11-generation-and-sampling.md) pour choisir les prochains IDs avec greedy, température, top-k et top-p, puis les décoder en texte.
