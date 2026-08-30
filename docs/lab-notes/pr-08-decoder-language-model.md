# Laboratoire PR08 - Instancier le modèle decoder-only complet

Date : 2026-08-30<br>
Branche : `feature/pr08-decoder-language-model`

## Objectifs

Ce laboratoire permet de :

- exécuter `[B,T] -> [B,T,V]` sur le vrai preset `mini-17m`;
- distinguer logits, probabilités et prédiction;
- prouver que le LM head partage réellement le poids d'embedding;
- comparer les paramètres DJL réels au calcul de PR01;
- mesurer le coût exact du mode non lié;
- vérifier la fermeture des huit caches RoPE.

## Exécuter le modèle lié

```powershell
.\gradlew.bat run --args="model forward --config configs/mini-17m.yaml --batch-size 1 --sequence-length 4 --seed 42"
```

Cette commande initialise bien les huit blocs et les 17,3 M poids. La séquence courte réduit seulement les activations et le calcul du forward; elle ne réduit pas les paramètres.

## Sortie observée

```text
Token IDs shape:         (1, 4) = [B, T]
Embedding shape:         (1, 4, 384) = [B, T, C]
Transformer blocks:      8
Final RMSNorm shape:     (1, 4, 384) = [B, T, C]
Logits shape:            (1, 4, 8192) = [B, T, V]
Logits finite:           true
Weight tying configured: true
Same Parameter object:   true
Same NDArray object:     true
Parameter tensors:       74
Actual parameters:       17,308,032
Theoretical parameters:  17,308,032
Counts match:            true
RoPE caches open:        8
Manager closed:          true
```

Les deux lignes `Same ... object` sont la preuve du partage. Une simple comparaison numérique pourrait être vraie juste après l'initialisation tout en devenant fausse après le premier update.

## Expérience 1 - Désactiver le weight tying

```powershell
.\gradlew.bat run --args="model forward --config configs/mini-17m.yaml --batch-size 1 --sequence-length 4 --seed 42 --untie-embeddings"
```

Résultat attendu :

```text
Weight tying configured: false
Same Parameter object:   false
Same NDArray object:     false
Parameter tensors:       75
Actual parameters:       20,453,760
Theoretical parameters:  20,453,760
Counts match:            true
```

La différence vaut :

```text
20 453 760 - 17 308 032 = 3 145 728
8192 × 384               = 3 145 728
```

Nous avons ajouté une seule matrice, mais elle contient plus de trois millions de poids. Le nombre de tenseurs seul ne permet donc jamais d'estimer la taille d'un modèle.

## Expérience 2 - Changer la longueur de séquence

```powershell
.\gradlew.bat run --args="model forward --config configs/mini-17m.yaml --batch-size 1 --sequence-length 8 --seed 42"
```

Les logits deviennent `[1,8,8192]`, mais le compte demeure `17 308 032`. Les poids dépendent de `V`, `C`, `F` et `N`; la longueur augmente les activations et le coût quadratique de l'attention.

## Expérience 3 - Lire un logit correctement

Pour chaque position `t`, le modèle produit 8192 scores. PR08 ne choisit pas encore le token suivant et n'affiche pas une probabilité. Conceptuellement :

```text
logits[0,t,:]             8192 scores réels
softmax(logits[0,t,:])    distribution éventuelle
argmax(logits[0,t,:])     choix greedy éventuel
```

PR09 utilisera les logits avec les vraies cibles next-token. PR11 ajoutera les stratégies de sélection pour la génération.

## Expérience 4 - Provoquer une limite de contexte

```powershell
.\gradlew.bat run --args="model forward --config configs/mini-17m.yaml --sequence-length 257"
```

Le preset accepte au maximum 256 positions. L'application retourne 2 et explique que la séquence dépasse `model.contextLength`; Gradle signale ensuite l'échec attendu du programme lancé.

## Exécuter les tests

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.model.LanguageModelHeadTest"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.model.DecoderLanguageModelTest"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.cli.ModelForwardCommandTest"
.\gradlew.bat check
```

Les tests protègent :

- multiplication contrôlée du head par la transposée de l'embedding;
- identité du `Parameter` et du `NDArray` liés;
- matrice indépendante `[C,V]` en mode non lié;
- formes embedding, blocs, norme finale et logits;
- valeurs et gradients finis pour tous les paramètres enregistrés;
- causalité des logits de bout en bout;
- compte réel et théorique des modèles tiny liés/non liés;
- allocation réelle de `mini-17m` avec exactement `17 308 032` poids;
- 74 tenseurs liés, 75 non liés et huit caches RoPE;
- libération de tous les caches sans fermeture prématurée du manager parent.

## Questions et réponses

### Pourquoi transposer l'embedding?

Le lookup lit des lignes d'une matrice `[V,C]`. Le head reçoit un vecteur de longueur `C` et doit produire `V` scores; la multiplication demande donc `[C,V]`, exactement la transposée.

### La transposée copie-t-elle trois millions de valeurs?

DJL/PyTorch représente normalement la transposée comme une vue avec une autre organisation logique. Surtout, aucun nouveau `Parameter` n'est enregistré : l'optimizer ne voit qu'un poids partagé.

### Pourquoi conserver un RMSNorm final?

Après le dernier résidu, l'échelle des états peut varier. La norme finale prépare une représentation stable pour le head et ajoute seulement `C=384` poids au preset.

### Les logits sont-ils déjà du texte?

Non. Ils indexent le vocabulaire du tokenizer. Il faudra choisir un ID, le décoder, puis répéter le forward pour générer du texte; ce chemin arrive en PR11.

### Pourquoi le modèle n'apprend-il rien dans cette commande?

Le forward calcule des scores avec des poids initialisés, mais aucune cible, loss ou mise à jour n'est fournie. PR09 ajoutera ces trois éléments et devra prouver qu'un tiny model peut surapprendre un batch.

### Pourquoi refuser un dropout non nul?

Ignorer silencieusement la configuration rendrait les expériences trompeuses. Tant que le dropout et son comportement train/eval ne sont pas implémentés et testés, le modèle exige `0.0`.

## Prochaine étape

PR09 construira la cross-entropy next-token, le backward, AdamW, le clipping, l'accumulation et les schedules. La première preuve ne sera pas un long entraînement : ce sera la chute nette de la loss sur un seul batch.
