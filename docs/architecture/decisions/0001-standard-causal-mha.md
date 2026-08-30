# ADR 0001 - Attention multi-tête causale standard

- Statut : accepté
- Date : 2026-08-30
- Portée : PR06

## Contexte

Le projet doit enseigner le chemin complet entre des états `[B,T,C]` et une sortie d'attention de même forme. GQA, MQA, Flash Attention et les kernels fusionnés masqueraient des étapes que nous voulons observer.

## Décision

Nous utilisons une self-attention multi-tête standard avec :

- quatre matrices carrées sans biais `Wq`, `Wk`, `Wv`, `Wo`, chacune `[C,C]`;
- `H` têtes de dimension `D=C/H`;
- RoPE appliqué à Q et K, jamais à V;
- scores divisés par `sqrt(D)`;
- masque additif `-Infinity` sur `keyPosition > queryPosition` avant softmax;
- calcul FP32 de référence, sans dropout dans PR06.

Le bloc possède exactement `4C²` paramètres. Pour `C=384`, cela donne `589 824` poids par couche d'attention.

## Pourquoi

Le facteur `1/sqrt(D)` empêche le produit scalaire de grandir mécaniquement avec la dimension et de pousser trop tôt le softmax vers des valeurs saturées. Le masque est appliqué avant softmax afin que les positions interdites reçoivent une probabilité exactement nulle après normalisation. RoPE modifie les relations positionnelles entre Q et K; V transporte le contenu à agréger et n'a pas besoin de cette rotation.

Des projections sans biais conservent le compte de paramètres annoncé depuis PR01. Une matrice causale explicite coûte davantage qu'un kernel fusionné, mais rend le mécanisme vérifiable et indépendant d'une primitive Transformer préfabriquée.

## Conséquences

- Les probabilités `[B,H,T,T]` peuvent être inspectées et testées directement.
- Le temps et la mémoire de l'attention croissent quadratiquement avec `T`.
- Le chemin de référence favorise la lisibilité, pas la vitesse maximale.
- Le KV cache, le dropout, mixed precision et les optimisations de kernel restent hors de PR06.

## Alternatives écartées maintenant

- **Attention non causale :** incorrecte pour la prédiction autoregressive du prochain token.
- **Masquage après softmax :** les lignes ne sommeraient plus à 1 sans renormalisation et le sens probabiliste serait brouillé.
- **GQA ou MQA :** utiles pour réduire le coût du KV cache, mais ajoutent une asymétrie pédagogique prématurée.
- **API Transformer DJL :** elle cacherait précisément Q/K/V, le masque et les changements de forme étudiés ici.
