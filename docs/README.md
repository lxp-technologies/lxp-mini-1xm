# Parcours d'apprentissage

Ce dossier accompagne le code dans l'ordre où les concepts sont construits. Ne lis pas seulement le résultat final : exécute l'expérience de chaque note de laboratoire avant de passer à la suivante.

## Commencer ici

1. [Objectifs du projet](00-project-goals.md)
2. [Architecture globale](architecture/overview.md)
3. [Compter les paramètres](architecture/parameter-counting.md)
4. [PR01 : fondations et configuration](lab-notes/pr-01-project-foundation.md)
5. [Comprendre la tokenization](02-tokenization.md)
6. [PR02 : byte tokenizer UTF-8](lab-notes/pr-02-byte-tokenizer.md)
7. [PR03 : entraîner et inspecter un byte-level BPE](lab-notes/pr-03-byte-level-bpe.md)
8. [Dataset, fenêtres et batches](04-dataset-and-sequences.md)
9. [PR04 : construire les séquences d'entraînement](lab-notes/pr-04-dataset-and-sequences.md)
10. [Embeddings, RMSNorm et RoPE](05-embeddings-rmsnorm-rope.md)
11. [Gestion de la mémoire native DJL](architecture/djl-memory-management.md)
12. [PR05 : premiers tenseurs et gradients](lab-notes/pr-05-embeddings-rmsnorm-rope.md)
13. [Self-attention causale](06-causal-self-attention.md)
14. [ADR : attention multi-tête causale standard](architecture/decisions/0001-standard-causal-mha.md)
15. [PR06 : inspecter et prouver la causalité](lab-notes/pr-06-causal-self-attention.md)
16. [SwiGLU et bloc Transformer pre-norm](07-transformer-block.md)
17. [ADR : bloc pre-norm avec SwiGLU](architecture/decisions/0002-prenorm-swiglu-block.md)
18. [PR07 : assembler et inspecter un bloc](lab-notes/pr-07-transformer-block.md)
19. [Modèle decoder-only et logits](08-decoder-language-model.md)
20. [ADR : weight tying réel et initialisation](architecture/decisions/0003-weight-tying-and-initialization.md)
21. [PR08 : instancier le modèle complet](lab-notes/pr-08-decoder-language-model.md)
22. [Plan complet des PR](plan-directeur.md)

Les prochains chapitres seront ajoutés avec leur implémentation. Un document ne prétendra jamais qu'un composant fonctionne avant que sa PR fournisse les tests correspondants.
