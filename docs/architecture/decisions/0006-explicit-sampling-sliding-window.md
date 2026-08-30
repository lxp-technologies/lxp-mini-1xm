# ADR 0006 - Sampling explicite avec fenêtre de contexte glissante

- Statut : accepté
- Date : 2026-08-30
- Portée : PR11

## Contexte

Le modèle produit des logits `[B,T,V]`, pas directement du texte. Il faut choisir un prochain ID, l'ajouter au contexte, recalculer les logits et recommencer. Cette étape doit rester inspectable et déterministe lorsqu'une seed est fixée.

Un KV cache accélérerait la génération, mais imposerait de modifier l'API de chaque bloc d'attention pour transporter, concaténer et tronquer les clés et valeurs de toutes les couches.

## Décision

PR11 sépare deux composants Kotlin sans dépendance DJL :

- `TokenSampler` applique greedy ou `température -> top-k -> softmax -> top-p -> renormalisation -> tirage`;
- `AutoregressiveGenerator` conserve tous les IDs produits, mais ne fournit au modèle que les `contextLength` derniers IDs.

La CLI garde l'adaptation DJL : elle extrait `logits[0, dernière_position, :]` dans un `FloatArray`, puis transmet ce tableau au sampler. Chaque étape affiche contexte, logit mis à l'échelle, probabilité, ID et pièce décodée.

La première version recalcule le forward complet de la fenêtre à chaque token. Elle n'utilise aucun KV cache.

## Pourquoi

Le découpage permet de tester les mathématiques sur de petits tableaux connus, sans backend natif. Le recalcul complet est plus lent, mais montre la vraie boucle autorégressive sans introduire simultanément une optimisation d'état complexe. La fenêtre glissante borne la mémoire et respecte la longueur maximale apprise par RoPE.

Le tokenizer est chargé depuis son artifact versionné et sa taille de vocabulaire doit correspondre à la configuration du checkpoint. Ce contrôle empêche de donner un autre sens aux mêmes IDs.

## Conséquences

- La génération est simple à tracer et les tests du sampler sont rapides.
- Une seed fixe reproduit les tirages du sampler pour les mêmes logits, dans la même implémentation Java.
- Greedy ne consomme pas la seed et choisit le plus petit ID en cas d'égalité.
- EOS est inclus dans les IDs générés, puis la boucle s'arrête immédiatement.
- Un prompt plus long que `contextLength` est accepté, mais seul son suffixe influence le prochain token.
- Le coût augmente fortement avec le nombre de tokens, car tout le contexte repasse dans le Transformer.
- Lorsque la fenêtre glisse, les positions RoPE sont recalculées à partir de zéro; cette simplification peut créer une rupture visible sur un tiny motif.

## Alternatives écartées

- **KV cache dès PR11 :** plus rapide, mais masque la boucle de base et multiplie les états et formes à valider.
- **Sampling dans DJL :** évite une copie CPU, mais rend les petits tests numériques et la seed plus dépendants du backend.
- **Un seul mode greedy :** déterministe, mais ne permet pas d'étudier diversité et filtrage.
- **Charger un tokenizer sans vérifier `vocabSize` :** risque de texte silencieusement faux.
- **Tronquer tout le résultat au contexte :** perd le texte complet alors que seule l'entrée du prochain forward doit être bornée.

## Preuves exigées

- argmax connu et égalité déterministe;
- température calculable, top-k et top-p renormalisés à 1;
- même séquence de tirages à seed fixe;
- arrêt EOS sans forward supplémentaire;
- suffixe glissant observé dans le provider de logits;
- commande complète chargeant checkpoint et tokenizer, puis produisant une trace décodable.
