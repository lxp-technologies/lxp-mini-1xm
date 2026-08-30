# Laboratoire PR11 - Des logits au texte

Date : 2026-08-30<br>
Branche : `feature/pr11-generation-sampling`

## Objectifs

Ce laboratoire permet de :

- entraîner le tiny model sur le motif encodé `abc `;
- recharger son dernier checkpoint avec le byte tokenizer compatible;
- lire chaque transition `contexte -> logit -> probabilité -> ID -> texte`;
- comparer quatre températures en gardant prompt, poids, seed et filtres constants;
- isoler ensuite l'effet de top-k et top-p;
- observer EOS, la seed et la fenêtre glissante.

Ce modèle n'a vu qu'un lot synthétique répété. Il démontre la mécanique de génération, pas une compétence linguistique.

## 1. Préparer un run neuf

Depuis PowerShell à la racine du dépôt :

```powershell
$runDir = "build/labs/pr11/demo-$(Get-Date -Format yyyyMMdd-HHmmss)"
.\gradlew.bat run --args="tokenizer byte create --output build/labs/pr11/tokenizer.json"
.\gradlew.bat run --args="train checkpoint-demo --config configs/lab-pr09-tiny.yaml --run-dir $runDir --before-updates 80 --after-updates 1"
```

Le nom horodaté est important : `checkpoint-demo` refuse volontairement un dossier existant. À 80 updates, la loss observée de référence est proche de `0,05`, mais elle peut varier selon l'environnement.

Le lot d'entraînement utilise les IDs byte suivants :

```text
texte : a   b   c  espace
IDs   : 100 101 102 35
```

## 2. Vérifier avec greedy

```powershell
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy greedy --show-candidates 4"
```

Le début attendu est :

```text
Prompt token IDs:   [100, 101, 102]
Strategy:           greedy
step=01 context=[100, 101, 102] chosen=35 piece=" " ...
step=02 context=[100, 101, 102, 35] chosen=100 piece="a" ...
step=03 context=[100, 101, 102, 35, 100] chosen=101 piece="b" ...
```

Sur le run de référence, le motif est correct jusqu'au glissement de la fenêtre de longueur 8, puis il peut diverger. Relance avec `--max-new-tokens 5` pour rester avant cette frontière.

## 3. Comparer quatre températures

Pour isoler la température, désactiver top-k et top-p. Conserver la même seed `42` et le même prompt :

```powershell
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy sample --temperature 0.25 --top-k 0 --top-p 1.0 --seed 42 --show-candidates 4"
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy sample --temperature 0.50 --top-k 0 --top-p 1.0 --seed 42 --show-candidates 4"
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy sample --temperature 1.00 --top-k 0 --top-p 1.0 --seed 42 --show-candidates 4"
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy sample --temperature 1.50 --top-k 0 --top-p 1.0 --seed 42 --show-candidates 4"
```

Consigner pour chaque run `Generated token IDs`, `Generated text` et la probabilité du premier choix :

| Température | Probabilité du premier choix | IDs | Observation |
|---:|---:|---|---|
| 0,25 | à relever | à relever | distribution très concentrée |
| 0,50 | à relever | à relever | plutôt conservatrice |
| 1,00 | à relever | à relever | distribution naturelle |
| 1,50 | à relever | à relever | alternatives amplifiées |

La comparaison est valide parce qu'une seule variable principale change. Une sortie identique à basse température est possible si le token dominant possède déjà presque toute la masse.

## 4. Isoler les filtres

Comparer d'abord sans filtre, puis avec les mêmes température et seed :

```powershell
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy sample --temperature 1.0 --top-k 0 --top-p 1.0 --seed 42 --show-candidates 8"
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --max-new-tokens 12 --strategy sample --temperature 1.0 --top-k 20 --top-p 0.9 --seed 42 --show-candidates 8"
```

`candidates` doit contenir au plus 20 entrées avant la limite d'affichage, puis seulement le noyau atteignant `0.9`. Les probabilités affichées sont renormalisées; leur somme interne vaut 1, ce que les tests unitaires vérifient.

## 5. Prouver la seed

Exécuter deux fois exactement la commande sans filtre à température `1.5`. Les `Generated token IDs` doivent être identiques. Remplacer ensuite `--seed 42` par `--seed 43`; une différence devient possible, mais n'est pas garantie si un candidat domine presque totalement.

Greedy reste identique quelle que soit la seed, puisqu'il n'effectue aucun tirage.

## 6. Observer la fenêtre

Dans chaque ligne, `context` grandit jusqu'à huit IDs. À l'étape suivante, l'ID le plus ancien disparaît :

```text
... context=[100, 101, 102, 35, 100, 101, 102, 35]
... context=[101, 102, 35, 100, 101, 102, 35, nouveau]
```

Le `Complete text` conserve pourtant tout le prompt et tous les tokens générés. Nous bornons l'entrée du modèle, pas le résultat utilisateur.

## 7. Provoquer des erreurs lisibles

Une température impossible :

```powershell
.\gradlew.bat run --args="generate --run-dir $runDir --tokenizer build/labs/pr11/tokenizer.json --prompt abc --strategy sample --temperature 0"
```

Un vocabulaire incompatible, en fournissant un BPE de taille différente, doit produire une erreur avant le forward. Un prompt vide est aussi refusé, sauf si `--add-bos` lui donne un token initial.

## Questions et réponses

### Pourquoi extraire seulement la dernière position des logits?

La position `t` prédit le token suivant son préfixe. Pour prolonger tout le prompt, seule la prédiction associée à son dernier token est pertinente. Les autres positions servent à la loss parallèle pendant l'entraînement.

### Pourquoi la température doit-elle être strictement positive?

Elle apparaît au dénominateur `logit / température`. À zéro, le calcul est indéfini. Pour obtenir un choix sans hasard, greedy exprime directement l'intention.

### Top-k et top-p sont-ils obligatoires ensemble?

Non. `top-k 0` et `top-p 1.0` les désactivent. Lorsqu'ils sont combinés, top-k limite d'abord le nombre de candidats, puis top-p conserve le plus petit noyau probabiliste dans cet ensemble.

### La seed rend-elle le modèle entièrement reproductible?

Elle rend le tirage du sampler reproductible pour une même suite de distributions. Un autre checkpoint, tokenizer, moteur ou device peut produire d'autres logits et donc une autre sortie malgré la même seed.

### Pourquoi ne pas ajouter un KV cache maintenant?

Le cache demanderait un état de clés et valeurs par couche et modifierait l'attention. PR11 privilégie une boucle correcte et visible; l'optimisation pourra être mesurée et ajoutée séparément après avoir une référence fiable.

### Pourquoi un token BPE isolé peut-il afficher `�`?

Le tokenizer est byte-level. Une pièce peut contenir seulement une partie des bytes d'un caractère UTF-8. Le décodage de la séquence complète rassemble ces bytes et constitue l'affichage de référence.

## Vérification automatique

```powershell
.\gradlew.bat test
```

Les tests protègent argmax, égalités, température calculable, filtres renormalisés, seed, EOS, fenêtre glissante, choix du loader byte/BPE et parcours CLI avec un vrai checkpoint.

## Critère de sortie

PR11 est réussie lorsque la commande permet de suivre sans étape cachée les derniers logits jusqu'aux IDs et au texte, tout en expliquant honnêtement la fenêtre glissante, la seed et l'absence de KV cache.
