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
- `generation` transforme depuis PR11 les derniers logits en token avec greedy ou sampling filtré, puis maintient la boucle autorégressive.
- `evaluation` calcule depuis PR12 validation loss, perplexité et débit sans backward ni modification des poids.
- `inference` possède depuis PR14 le modèle chargé, le tokenizer, les scopes et la limite de concurrence; PR15 y ajoute un cache KV isolé par requête, les politiques de contexte et les métriques prefill/decode, sans connaître HTTP.
- `cli` rend chaque concept exécutable depuis Gradle.
- `tokenizer` et `data` préparent les `IntArray`; `model` les convertira progressivement en calcul neuronal DJL.

## Décisions de PR01

- mono-module Gradle pour garder la navigation simple;
- YAML strict afin qu'une faute de frappe échoue immédiatement;
- projections linéaires futures sans biais;
- weight tying activé par défaut;
- JDK 25, Gradle 9.1+ et Kotlin/JVM;
- aucune dépendance DJL n'a été ajoutée avant le premier tenseur en PR05; le code du modèle dépend de l'API DJL et non des classes internes PyTorch.

## État après PR15

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
    LOAD --> LAST[Derniers logits<br/>V]
    LAST --> SAMPLE[Greedy ou sampling<br/>température, top-k, top-p]
    SAMPLE --> TOKEN[Prochain ID]
    TOKEN --> WINDOW[Fenêtre glissante]
    WINDOW --> EMB
    TOKEN --> DECODE[Texte décodé]
    LOAD --> EVAL[Évaluation sans gradient]
    VAL[Validation BPE<br/>checksum vérifié] --> EVAL
    EVAL --> METRICS[Loss, perplexité<br/>tokens/s]
    CKPT --> RUNTIME[InferenceRuntime<br/>chargé une fois]
    RUNTIME --> REQUEST[Scope DJL par requête<br/>concurrence sérialisée]
    REQUEST --> LAST
    REQUEST --> PREFILL[Prefill du prompt]
    PREFILL --> KV[Cache K/V par couche<br/>B x H x S x D]
    KV --> DECODE[Decode un token]
    DECODE --> LAST
```

Le forward, la boucle d'entraînement, les checkpoints, la génération et l'évaluation de corpus existent. PR15 conserve K/V par couche pendant une requête, distingue prefill et decode, puis ferme cet état avec son scope. La reprise AdamW exacte et une qualité suffisante pour lancer le 17 M restent hors périmètre.
