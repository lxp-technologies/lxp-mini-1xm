# ADR 0010 - Cache KV isolé par requête

## Statut

Accepté le 2026-08-30 pour PR15.

## Contexte

PR14 recalculait toutes les couches sur toute la fenêtre avant chaque token. Les clés et valeurs passées sont invariantes en attention causale, mais leur conservation introduit un état natif mutable. Cet état ne peut appartenir au modèle partagé sans risquer qu'une requête lise le contexte d'une autre.

RoPE lie chaque clé à une position. La fenêtre glissante PR11 renumérote les tokens conservés après éviction; retirer seulement le premier K/V produirait donc une sémantique différente du recalcul complet.

## Décision

- Le forward d'entraînement complet reste inchangé.
- Le decoder expose un forward incrémental inference-only.
- Chaque couche possède K et V de forme `[B,H,S,D]` dans un `DecoderKeyValueCache`.
- Le cache est créé sous le manager de requête et fermé avec lui.
- Le prefill traite un chunk; le decode traite ensuite un token.
- `REJECT` valide le budget complet avant calcul.
- `SLIDING_WINDOW` tronque à gauche explicitement, puis invalide et reconstruit le cache lorsque la fenêtre glisse.
- Le runtime reste sérialisé; les caches demeurent malgré tout isolés par appel.
- Les métriques distinguent temps et tokens prefill/decode.

## Conséquences

Le nombre de tokens projetés devient linéaire dans la longueur générée tant que la fenêtre ne glisse pas. La mémoire du cache croît avec `2 × layers × B × H × S × D × bytesParValeur`.

À fenêtre pleine, `SLIDING_WINDOW` paie un nouveau prefill pour conserver l'équivalence PR11. `REJECT` est plus simple et convient aux benchmarks de débit. Le cache ajoute un coût fixe qui peut rendre un très petit décodage plus lent.

## Alternatives rejetées

- Un cache global au runtime fuiterait le contexte entre requêtes.
- Un cache par modèle ou par thread aurait un ownership ambigu et pourrait survivre à la requête.
- Évincer K/V sans les reroter ne correspondrait plus au recalcul avec positions RoPE remises à zéro.
- Étendre implicitement les positions RoPE au-delà de `contextLength` annoncerait une capacité non entraînée et non évaluée.
- Remplacer le forward complet supprimerait notre oracle et risquerait de modifier l'entraînement.
