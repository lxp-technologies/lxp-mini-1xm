# PR04 - Dataset et séquences

## Ce que nous avons construit

PR04 lit un corpus par blocs, préserve exactement la tokenization BPE aux frontières, compte les tokens, crée deux plages sans recouvrement, produit des fenêtres next-token et regroupe celles-ci en batches primitifs reproductibles.

## Expérience 1 - Voir le décalage d'un token

```powershell
.\gradlew.bat run --args="dataset window --tokens 10,20,30,40,50 --context-length 4"
```

Résultat attendu :

```text
input:  [10, 20, 30, 40]
target: [20, 30, 40, 50]
```

Les deux formes sont `[T] = [4]`. La fenêtre source a une longueur `T+1 = 5`.

## Expérience 2 - Préparer le tokenizer du laboratoire

```powershell
.\gradlew.bat run --args="tokenizer bpe train --input docs/lab-notes/samples/pr03-corpus.txt --vocab-size 272 --output build/labs/pr04/tokenizer.json"
```

Cette commande réutilise le petit corpus versionné afin que l'expérience reste rapide. Elle démontre la mécanique, mais ne constitue donc pas une mesure de validation indépendante : le tokenizer a déjà vu tous ces bytes. Pour un véritable projet, entraîne le tokenizer sur la partie train préparée seulement, puis fige son JSON avant de toucher à la validation.

## Expérience 3 - Inspecter le pipeline complet

```powershell
.\gradlew.bat run --args="dataset inspect --corpus docs/lab-notes/samples/pr03-corpus.txt --tokenizer build/labs/pr04/tokenizer.json --context-length 8 --batch-size 3 --validation-fraction 0.2 --split train --shuffle-buffer 4 --seed 42 --show-batches 2"
```

Résultats observés le 2026-08-29 :

| Mesure | Valeur |
|---|---:|
| Tokens totaux | 181 |
| Plage train | `[0,145)` = 145 tokens |
| Plage validation | `[145,181)` = 36 tokens |
| Fenêtres train complètes | 18 |
| Tokens train finaux inutilisés | 0 |
| Batches planifiés | 6 de forme `[3,8]` |

Dans chaque ligne affichée, vérifie que `target[0] == input[1]`, puis que cette égalité continue jusqu'à la dernière position disponible.

## Expérience 4 - Inspecter la validation séparément

```powershell
.\gradlew.bat run --args="dataset inspect --corpus docs/lab-notes/samples/pr03-corpus.txt --tokenizer build/labs/pr04/tokenizer.json --context-length 8 --batch-size 3 --validation-fraction 0.2 --split validation --shuffle-buffer 0 --show-batches 2"
```

La plage doit être exactement `[145,181)`. Elle contient 4 fenêtres complètes et 3 tokens finaux qui ne suffisent pas à former une autre fenêtre de 9 tokens. Aucun contexte train n'est repris pour compléter le début de validation.

## Expérience 5 - Prouver le shuffle reproductible

Exécute deux fois la commande de l'expérience 3 avec `--seed 42`: les batches affichés doivent être identiques. Remplace ensuite la seed par 43 : l'ordre doit changer, tandis que le nombre de tokens, les plages et le nombre de fenêtres restent identiques.

Le test automatisé ciblé est :

```powershell
.\gradlew.bat test --tests '*TokenPipelineTest.bounded shuffle*'
```

## Expérience 6 - Tester les frontières de streaming

```powershell
.\gradlew.bat test --tests '*StreamingBpeTokenReaderTest*'
```

Le test relit le même texte avec des blocs de 1 byte, 2 bytes, et ainsi de suite. Chaque résultat doit être identique à l'encodage complet, même lorsqu'un bloc coupe un caractère multibyte ou un merge appris.

## Questions de compréhension

1. Pourquoi faut-il `T+1` tokens pour créer un input et une target de longueur `T`?
2. Pourquoi construisons-nous les fenêtres après le split?
3. Pourquoi le lecteur BPE conserve-t-il un suffixe entre deux blocs?
4. Pourquoi faire deux passages au lieu de conserver tous les IDs?
5. Quelle différence y a-t-il entre les trailing tokens et un dernier batch partiel?
6. Pourquoi une seed ne suffit-elle pas à décrire un shuffle reproductible?

## Réponses et explications

### 1. Pourquoi `T+1`?

L'input retire le dernier token et la target retire le premier. Les deux séquences ont alors `T` positions alignées, et chaque target est exactement le prochain token de l'input.

### 2. Pourquoi splitter avant les fenêtres?

Si une fenêtre chevauchait la frontière, certains tokens serviraient à la fois à l'entraînement et à la validation. Construire chaque lecteur sur une plage disjointe interdit mécaniquement ce partage.

### 3. Pourquoi conserver un suffixe?

Une pièce BPE peut commencer dans un bloc et finir dans le suivant. Le suffixe contient tous les bytes encore susceptibles de participer à une future pièce; le préfixe plus ancien peut être émis sans changer le résultat.

### 4. Pourquoi deux passages?

Le premier calcule le nombre exact de tokens nécessaire au split. Le second ne garde que les petits buffers requis pour les fenêtres et batches. Nous payons une lecture supplémentaire pour ne pas payer une mémoire proportionnelle au corpus.

### 5. Fragment final ou batch partiel?

Un fragment de moins de `T+1` tokens ne peut produire aucune fenêtre et est toujours ignoré. Un batch partiel contient au contraire des fenêtres complètes, mais en nombre inférieur à `B`; il est conservé par défaut et supprimé seulement avec `--drop-last-batch`.

### 6. Pourquoi la seed ne suffit-elle pas?

L'ordre dépend aussi du corpus, du tokenizer, du split, de `T`, du stride, de la taille du buffer et de l'algorithme de PR04. Toute expérience doit consigner ces valeurs avec la seed.

## Exécuter toute la validation

```powershell
.\gradlew.bat check
```

Sous Linux ou macOS, remplace `.\gradlew.bat` par `./gradlew`.

## Nettoyer les artefacts

```powershell
Remove-Item -Recurse -Force build/labs/pr04
```

## Ce que la prochaine PR ajoutera

PR05 convertira les IDs `[B,T]` en vecteurs `[B,T,C]`, puis introduira RMSNorm et RoPE avec DJL. PR04 ne crée encore aucun tenseur.
