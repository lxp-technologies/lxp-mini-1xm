# Architecture globale

```mermaid
flowchart LR
    TXT[Texte UTF-8] --> TOK[Tokenizer maison]
    TOK --> IDS[IDs B x T]
    IDS --> MODEL[Decoder-only Transformer]
    MODEL --> LOGITS[Logits B x T x V]
    LOGITS --> NEXT[Prochain token]
    NEXT --> TOK
```

## Frontières

- `config` charge et valide les expériences sans dépendre d'une bibliothèque neuronale.
- `model` contient le compteur théorique et les primitives différentiables. PR05 ajoute embedding, RMSNorm et RoPE; PR06 ajoute l'attention; PR07 assemble un bloc; PR08 construit le decoder complet et ses logits.
- `training` relie depuis PR09 les logits aux targets avec cross-entropy, backward, accumulation, clipping global, AdamW et warmup/cosine.
- `checkpoint` crée depuis PR10 les runs, manifestes, checksums, métriques et round-trips de poids.
- `cli` rend chaque concept exécutable depuis Gradle.
- `tokenizer` et `data` préparent les `IntArray`; `model` les convertira progressivement en calcul neuronal DJL.
- les futurs packages `generation` et `evaluation` apparaîtront seulement dans leur PR.

## Décisions de PR01

- mono-module Gradle pour garder la navigation simple;
- YAML strict afin qu'une faute de frappe échoue immédiatement;
- projections linéaires futures sans biais;
- weight tying activé par défaut;
- JDK 25, Gradle 9.1+ et Kotlin/JVM;
- aucune dépendance DJL n'a été ajoutée avant le premier tenseur en PR05; le code du modèle dépend de l'API DJL et non des classes internes PyTorch.

## État après PR10

```mermaid
flowchart LR
    IDS[IDs<br/>B x T] --> EMB[TokenEmbedding]
    EMB --> BLOCKS[N TransformerBlock]
    BLOCKS --> NORM[RMSNorm final]
    NORM --> HEAD[LanguageModelHead]
    HEAD --> LOGITS[Logits<br/>B x T x V]
    LOGITS --> LOSS[Cross-entropy]
    TARGETS[Targets<br/>B x T] --> LOSS
    LOSS --> BACK[Backward]
    BACK --> OPT[Clip global + AdamW]
    OPT -. met à jour .-> EMB
    EMB -. même poids .-> HEAD
    OPT --> CKPT[Checkpoint versionné]
    CKPT --> LOAD[Nouveau modèle]
    LOAD -. poids identiques .-> EMB
```

Le forward, la boucle d'entraînement et le checkpoint de poids existent. PR10 prouve des logits identiques dans un nouveau modèle et restaure la progression, mais déclare la reprise d'entraînement non exacte sans moments AdamW ni état RNG. La génération arrive en PR11 et l'évaluation en PR12.
