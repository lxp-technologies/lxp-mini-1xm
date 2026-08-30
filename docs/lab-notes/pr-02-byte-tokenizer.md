# PR02 - Byte tokenizer UTF-8

## Ce que nous avons construit

Un tokenizer Kotlin capable de convertir tout texte UTF-8 valide en IDs et de reconstruire exactement le texte. Il réserve `PAD=0`, `BOS=1`, `EOS=2`, représente les 256 bytes par les IDs `3..258`, sauvegarde un artefact JSON versionné et fournit deux commandes CLI.

## Ce qu'il faut comprendre

Le modèle ne verra jamais directement une lettre ou un emoji. Il verra des IDs. Dans cette PR, chaque ID ordinaire correspond à un byte UTF-8, de sorte qu'un caractère visible peut produire un, deux, trois ou quatre tokens.

## Formules importantes

```text
tokenId = unsignedByte + 3
unsignedByte = tokenId - 3
vocabularySize = 3 special tokens + 256 bytes = 259
```

## Tensor shapes importantes

Il n'y a pas encore de tenseur :

```text
String -> ByteArray[T] -> IntArray[T]
String -> ByteArray[T] -> IntArray[T + 2] avec BOS et EOS
```

La future forme `[B, T]` arrivera avec le dataset et le batching en PR04.

## Comment l'exécuter

Sous Windows PowerShell, depuis la racine du projet :

```powershell
.\gradlew.bat test
.\gradlew.bat run --args="tokenizer byte inspect --text-file docs/lab-notes/samples/pr02-unicode.txt --add-bos --add-eos"
```

Sous Linux ou macOS :

```bash
./gradlew test
./gradlew run --args="tokenizer byte inspect --text-file docs/lab-notes/samples/pr02-unicode.txt --add-bos --add-eos"
```

Le fichier est utilisé afin que l'emoji traverse intact les terminaux Windows dont la page de code ne prend pas complètement en charge Unicode.

## Expériences à essayer

### Expérience 1 - Comparer ASCII, accent et emoji

```powershell
.\gradlew.bat run --args="tokenizer byte inspect --text A"
.\gradlew.bat run --args="tokenizer byte inspect --text-file docs/lab-notes/samples/pr02-unicode.txt"
```

Résultats attendus :

| Texte | Bytes UTF-8            | Token IDs              |
|-------|------------------------|------------------------|
| `A`   | `[65]`                 | `[68]`                 |
| `é`   | `[195, 169]`           | `[198, 172]`           |
| `👋`  | `[240, 159, 145, 139]` | `[243, 162, 148, 142]` |

Observe que le décalage est toujours exactement 3.

L'échantillon contient `Aé👋` suivi du newline final du fichier. Tu verras donc aussi le byte `10`, qui devient le token ID `13`.

### Expérience 2 - Rendre BOS et EOS visibles

```powershell
.\gradlew.bat run --args="tokenizer byte inspect --text A --add-bos --add-eos"
```

Résultat attendu : les token IDs sont `[1, 68, 2]`. Le texte décodé reste `A`, car les tokens spéciaux sont ignorés lors de la reconstruction du texte.

### Expérience 3 - Créer et lire `tokenizer.json`

```powershell
.\gradlew.bat run --args="tokenizer byte create --output build/labs/pr02/tokenizer.json"
Get-Content build/labs/pr02/tokenizer.json
```

Le JSON doit indiquer la version 1, le type `byte`, une taille de vocabulaire de 259, un décalage de 3 et les trois tokens spéciaux. Pour vérifier le round-trip de sauvegarde/chargement :

```powershell
.\gradlew.bat test --tests '*ByteTokenizerArtifactStoreTest'
```

### Expérience 4 - Exécuter seulement les propriétés Unicode

```powershell
.\gradlew.bat test --tests '*ByteTokenizerTest'
```

Ce groupe vérifie notamment français, emoji, caractères asiatiques, Unicode combiné, texte vide et les 256 valeurs de byte.

### Nettoyer l'artefact de laboratoire

```powershell
Remove-Item -Recurse -Force build/labs/pr02
```

Selon la police et la page de code du terminal, le glyphe de l'emoji peut être mal dessiné. Les lignes `UTF-8 bytes` et `Token IDs`, ainsi que les tests de round-trip, restent la vérification fiable de son intégrité.

## Résultats attendus

- `encode` puis `decode` restitue exactement chaque texte UTF-8 valide;
- les 256 bytes ont chacun un ID distinct;
- aucune entrée ne produit `<unk>`;
- les tokens spéciaux ne peuvent pas entrer en collision avec un byte;
- un artefact incompatible ou une séquence UTF-8 invalide est rejeté clairement.

## Questions de compréhension

1. Pourquoi le byte tokenizer n'a-t-il pas besoin de `<unk>`?
2. Pourquoi `é` produit-il deux tokens et `👋` quatre?
3. Pourquoi ajoutons-nous 3 à chaque valeur de byte?
4. Quelle différence y a-t-il entre `EOS` et `PAD`?
5. Si tous les bytes sont représentables, pourquoi certaines suites de tokens ne sont-elles pas du texte UTF-8 valide?

## Réponses et explications

### 1. Pourquoi n'y a-t-il pas de `<unk>`?

Tout texte UTF-8 est composé uniquement de bytes ayant une valeur entre 0 et 255. Comme les 256 valeurs ont chacune un token, aucune nouvelle lettre, langue ou emoji ne peut introduire un byte inconnu.

### 2. Pourquoi certains caractères produisent-ils plusieurs tokens?

UTF-8 utilise une longueur variable. ASCII tient généralement dans un byte, `é` en demande deux et `👋` quatre. Notre tokenizer produit un token par byte, pas un token par caractère visible.

### 3. Pourquoi ajouter 3?

Les IDs 0, 1 et 2 sont déjà réservés à `PAD`, `BOS` et `EOS`. Ajouter 3 déplace les bytes dans l'intervalle `3..258` et garantit qu'aucun byte ne partage l'ID d'un token spécial.

### 4. Quelle différence y a-t-il entre `EOS` et `PAD`?

`EOS` fait partie du sens de la séquence : il annonce que le texte est terminé. `PAD` est seulement un remplissage technique destiné à aligner plus tard plusieurs séquences dans un batch. Il ne signifie pas que le modèle a choisi de terminer.

### 5. Pourquoi une suite de bytes peut-elle être invalide en UTF-8?

La représentation de chaque byte et la validité de leur combinaison sont deux propriétés différentes. Certains bytes annoncent une séquence de deux à quatre éléments et doivent être suivis de bytes de continuation précis. `decodeToBytes()` peut reconstruire toute suite; `decode()` vérifie en plus sa grammaire UTF-8.

## Ce que la prochaine PR ajoutera

PR03 apprendra un byte-level BPE en Kotlin. Il commencera avec ces mêmes 256 bytes, comptera les paires adjacentes fréquentes et créera de nouveaux tokens capables de représenter plusieurs bytes à la fois.
