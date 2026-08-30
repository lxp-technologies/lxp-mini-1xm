# Laboratoire PR06 - Inspecter et prouver la causalité

Date : 2026-08-30<br>
Branche : `feature/pr06-causal-self-attention`

## Objectifs

À la fin du laboratoire, nous pouvons :

- suivre les formes de Q, K, V et des probabilités;
- lire une matrice d'attention de quatre tokens;
- vérifier que chaque ligne est une distribution;
- démontrer qu'un token futur ne modifie pas le passé;
- vérifier que les gradients traversent les quatre projections.

## Exécuter PR06

Depuis la racine du dépôt, sous PowerShell :

```powershell
.\gradlew.bat run --args="model attention --d-model 8 --num-heads 2 --sequence-length 4 --context-length 16 --seed 42"
```

Au premier usage de DJL sur une nouvelle machine, le moteur peut préparer ses bibliothèques natives. Le laboratoire force le CPU et utilise FP32 afin de garder une référence facile à reproduire.

## Sortie observée

Les valeurs exactes dépendent de l'initialisation et du moteur, mais les invariants restent identiques :

```text
Input shape:            (1, 4, 8) = [B, T, C]
Q/K/V shape:            (1, 2, 4, 4) = [B, H, T, D]
Attention shape:        (1, 2, 4, 4) = [B, H, T, T]
Output shape:           (1, 4, 8) = [B, T, C]
Attention parameters:   256
Head 0 attention:
  [1.000000, 0.000000, 0.000000, 0.000000]
  [0.539409, 0.460591, 0.000000, 0.000000]
  [0.301707, 0.401081, 0.297212, 0.000000]
  [0.201040, 0.361776, 0.234494, 0.202690]
Head 0 row sums:        [1.000000, 1.000000, 1.000000, 1.000000]
Future probability max: 0.000000
Past output max delta:  0.000000
Manager closed:         true
```

La première ligne vaut toujours `[1,0,0,0]`, car la requête 0 ne possède qu'une clé permise. Les autres valeurs ne sont pas des règles à mémoriser : elles viennent des poids initialisés et des vecteurs d'entrée.

## Expérience 1 - Changer le nombre de têtes

Le nombre de paramètres reste `4C²`, mais la décomposition change :

```powershell
.\gradlew.bat run --args="model attention --d-model 16 --num-heads 2 --sequence-length 4 --context-length 16 --seed 42"
.\gradlew.bat run --args="model attention --d-model 16 --num-heads 4 --sequence-length 4 --context-length 16 --seed 42"
```

Dans les deux cas, les paramètres valent `4 × 16² = 1024`. La dimension par tête passe de 8 à 4 et les probabilités changent, car chaque tête compare un autre sous-espace.

## Expérience 2 - Voir le coût quadratique

```powershell
.\gradlew.bat run --args="model attention --d-model 8 --num-heads 2 --sequence-length 8 --context-length 16 --seed 42"
```

La forme des probabilités devient `(1,2,8,8)`. Passer de 4 à 8 tokens multiplie le nombre de cases par quatre, de `2×4×4=32` à `2×8×8=128`.

## Expérience 3 - Provoquer une validation utile

```powershell
.\gradlew.bat run --args="model attention --d-model 10 --num-heads 2 --sequence-length 4"
```

Ici `D=10/2=5`, ce qui est impair et incompatible avec les paires RoPE. L'application retourne le code 2; Gradle affiche donc aussi `FAILED` et retourne le code 1 parce que le programme lancé a volontairement refusé la configuration.

## Exécuter les preuves automatiques

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.model.CausalSelfAttentionTest"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.cli.ModelAttentionCommandTest"
.\gradlew.bat check
```

Les tests protègent :

- sortie `[B,T,C]` et probabilités `[B,H,T,T]`;
- sommes de lignes proches de 1;
- zéros exacts au-dessus de la diagonale;
- invariance de toutes les sorties passées quand le dernier token change;
- gradients finis pour l'entrée et les quatre poids `[C,C]`;
- `4C²` paramètres sans biais;
- fermeture du cache RoPE sans fermer prématurément le manager parent.

## Questions et réponses

### Pourquoi le softmax est-il calculé sur le dernier axe?

Pour chaque triplet `(batch, tête, requête)`, le dernier axe énumère les clés possibles. Le softmax transforme donc les scores de ces clés en une distribution qui somme à 1.

### Pourquoi Q et K reçoivent RoPE, mais pas V?

Q et K déterminent la compatibilité entre positions; les tourner modifie cette relation selon leur distance. V contient le signal finalement additionné. Le tourner changerait le contenu transporté plutôt que seulement la manière de choisir où lire.

### Est-ce que plusieurs têtes ajoutent des paramètres?

Pas lorsque `C` reste fixe dans cette architecture. Les matrices demeurent `[C,C]`; nous découpons seulement leur sortie en `H` groupes de taille `D=C/H`.

### Le masque causal suffit-il à prouver l'absence de fuite?

Le triangle vérifie directement les probabilités, mais une erreur de reshape ou de fusion pourrait encore mélanger les positions. C'est pourquoi le test anti-fuite vérifie aussi le comportement final de bout en bout.

### Pourquoi ne pas optimiser immédiatement la matrice `T×T`?

La version explicite établit d'abord une référence correcte et compréhensible. Une future optimisation devra produire les mêmes sorties dans les tolérances et conserver le test anti-fuite; sans cette référence, une accélération serait difficile à valider.

## Prochaine étape

PR07 ajoutera SwiGLU, deux RMSNorm et les connexions résiduelles autour de cette attention pour former un bloc Transformer pre-norm complet.
