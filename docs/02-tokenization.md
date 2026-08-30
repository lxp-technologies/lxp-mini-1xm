# Tokenization : des caractères aux bytes

## Qu'est-ce que c'est?

Un modèle ne reçoit pas directement du texte. Un tokenizer transforme le texte en une séquence d'entiers appelés **token IDs**. PR02 commence par le tokenizer le plus universel et le plus facile à inspecter : un token représente exactement un byte UTF-8.

```mermaid
flowchart LR
    TEXT[Texte Unicode] -->|encodage UTF-8| BYTES[Bytes 0..255]
    BYTES -->|+ 3| IDS[Token IDs 3..258]
    IDS -->|- 3| BYTES2[Bytes 0..255]
    BYTES2 -->|décodage UTF-8 strict| TEXT2[Texte Unicode]
```

## Pourquoi en avons-nous besoin?

Les couches neuronales manipulent des nombres. Les token IDs serviront plus tard d'indices dans la matrice d'embeddings. Le byte tokenizer fournit dès maintenant trois propriétés importantes :

- tout texte UTF-8 valide peut être encodé;
- la table de correspondance est fixe et entièrement explicable;
- aucun token `<unk>` n'est nécessaire, car tout fichier UTF-8 est une suite de bytes parmi les 256 valeurs possibles.

Il est moins compact qu'un tokenizer BPE. PR03 apprendra justement à fusionner des suites de bytes fréquentes afin de réduire le nombre de tokens.

## Quelle est la formule?

Nous réservons les trois premiers IDs :

```text
PAD = 0
BOS = 1
EOS = 2
```

Chaque byte non signé `b` compris entre 0 et 255 utilise ensuite :

```text
tokenId = b + 3
b = tokenId - 3
```

Le vocabulaire contient donc `3 + 256 = 259` IDs. Par exemple, `A` vaut 65 en UTF-8 et devient le token ID 68.

## Caractère, code point, byte et token

Ces notions ne sont pas interchangeables. Un caractère visible peut demander plusieurs bytes UTF-8 :

| Texte | Bytes UTF-8            | Token IDs              |
|-------|------------------------|------------------------|
| `A`   | `[65]`                 | `[68]`                 |
| `é`   | `[195, 169]`           | `[198, 172]`           |
| `👋`  | `[240, 159, 145, 139]` | `[243, 162, 148, 142]` |

Ainsi, le byte tokenizer produit un token par byte, pas un token par lettre ni par caractère affiché.

## Quelles sont les dimensions?

PR02 ne crée toujours aucun tenseur. Pour un texte encodé en `T` bytes :

```text
texte                 String
bytes UTF-8           ByteArray[T]
token IDs             IntArray[T]
token IDs avec BOS/EOS IntArray[T + 2]
```

Plus tard, un batch de séquences aura une forme `[B, T]`. `PAD` permettra d'aligner des séquences de longueurs différentes, mais PR02 n'effectue pas encore de batching.

## À quoi servent les tokens spéciaux?

- `BOS` marque explicitement le début d'une séquence;
- `EOS` marque sa fin et permettra à la génération de savoir quand s'arrêter;
- `PAD` remplira plus tard les positions inutilisées d'un batch.

Ils sont optionnels lors de l'encodage. Le décodage normal les ignore afin de reconstruire seulement le texte original.

## Pourquoi le décodage UTF-8 est-il strict?

Tous les bytes sont représentables individuellement, mais toute suite arbitraire de bytes ne forme pas nécessairement un texte UTF-8 valide. Par exemple, le byte `195` annonce le début d'une séquence de deux bytes; seul, il est incomplet.

`decodeToBytes()` peut toujours reconstruire les bytes. `decode()` exige qu'ils forment un texte UTF-8 valide et signale sinon une erreur. Cette distinction évite une corruption silencieuse par un caractère de remplacement.

## Où est-ce implémenté?

- `ByteTokenizer.kt` contient les formules encode/decode;
- `SpecialToken.kt` définit `PAD`, `BOS` et `EOS`;
- `ByteTokenizerArtifactStore.kt` sauvegarde et valide `tokenizer.json`;
- `ByteTokenizerCommands.kt` rend le mécanisme observable depuis la CLI.

## Quels tests prouvent que cela fonctionne?

`ByteTokenizerTest` couvre ASCII, français, emoji, caractères asiatiques, Unicode combiné, espaces, newlines, texte vide, les 256 bytes, les tokens spéciaux et les erreurs UTF-8. `ByteTokenizerArtifactStoreTest` protège le format versionné et `ByteTokenizerCommandsTest` vérifie les commandes exécutables.

Le test des 256 valeurs est la preuve essentielle qu'aucun byte inconnu ne peut apparaître.

## Qu'est-ce qui casse si cette partie est incorrecte?

- un décalage incorrect change le sens de tous les IDs;
- un encode/decode non symétrique corrompt le corpus et les générations;
- une collision avec `PAD/BOS/EOS` rend les séquences ambiguës;
- une sérialisation non versionnée peut associer un checkpoint au mauvais vocabulaire;
- un traitement par caractères au lieu de bytes échoue sur les accents ou emoji.

## Expérience à faire

Suis la [note de laboratoire PR02](lab-notes/pr-02-byte-tokenizer.md). Elle montre comment inspecter plusieurs écritures, ajouter les tokens spéciaux, créer l'artefact JSON et exécuter uniquement les tests du tokenizer.
