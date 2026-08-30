# ADR 0002 - Bloc pre-norm avec SwiGLU

- Statut : accepté
- Date : 2026-08-30
- Portée : PR07

## Contexte

PR06 fournit l'attention causale, mais un Transformer exige aussi une transformation positionnelle indépendante et des chemins résiduels. Nous devons choisir l'emplacement des normalisations et la famille du feed-forward sans cacher ces mécanismes derrière un bloc préfabriqué.

## Décision

Chaque bloc utilise cette structure :

```text
a = x + Attention(RMSNorm(x))
y = a + SwiGLU(RMSNorm(a))
```

SwiGLU utilise trois matrices sans biais :

```text
gate   = x Wgate
value  = x Wvalue
hidden = SiLU(gate) × value
output = hidden Wdown
```

`Wgate` et `Wvalue` ont la forme `[C,F]`; `Wdown` a la forme `[F,C]`. Le feed-forward contient donc `3CF` paramètres et le bloc complet `4C² + 3CF + 2C`.

## Pourquoi

Pre-norm offre un chemin résiduel direct qui ne traverse pas obligatoirement la normalisation ou la sous-couche. Cela rend le flux du gradient plus facile à raisonner lorsqu'on empile des blocs. Les résidus permettent aussi à une sous-couche initialement peu utile de modifier l'état plutôt que de devoir le reconstruire entièrement.

SwiGLU sépare une branche de contrôle non linéaire et une branche de contenu. Il est plus riche qu'un simple `ReLU(XW1)W2`, tout en restant calculable explicitement. Les projections sans biais conservent le nombre exact de paramètres prévu par le projet.

## Conséquences

- Deux vecteurs RMSNorm `[C]` ajoutent `2C` poids par bloc.
- Les formes externes restent `[B,T,C]`; l'intermédiaire SwiGLU vaut `[B,T,F]`.
- La voie résiduelle ajoute une contribution identité au Jacobien, mais la norme totale du gradient n'est pas garantie supérieure à cause des directions et annulations possibles.
- Trois matrices FFN coûtent davantage que deux matrices ReLU.
- Le dropout demeure absent de la référence PR07.

## Alternatives écartées maintenant

- **Post-norm :** valide historiquement, mais place la normalisation après l'addition et rend le chemin identité moins direct.
- **ReLU/GELU à deux matrices :** plus simple, mais ne correspond pas à l'architecture et au compte de paramètres retenus.
- **Biais appris :** ajoutent des paramètres sans être nécessaires à l'objectif pédagogique actuel.
- **Désactivation configurable des résidus en production :** refusée; seul `forwardForInspection` expose cette expérience, tandis que le forward normal les garde toujours actifs.
