# Compter les paramètres

## Qu'est-ce qu'un paramètre?

Un paramètre est une valeur apprise pendant l'entraînement. Une matrice de forme `[a, b]` contient `a × b` paramètres. Les activations temporaires et les tokens du contexte ne sont pas des paramètres.

## Formule de `mini-17m`

Avec `V=8192`, `C=384`, `N=8` et `F=1024` :

```text
embeddings       = V × C             =  3 145 728
attention        = N × 4 × C²        =  4 718 592
SwiGLU           = N × 3 × C × F     =  9 437 184
RMSNorm          = N × 2 × C + C     =      6 528
LM head partagé  =                            0
total            =                       17 308 032
```

Les quatre matrices d'attention sont `Wq`, `Wk`, `Wv` et `Wo`. Les trois matrices SwiGLU sont gate, value et down. Chaque bloc possède deux vecteurs RMSNorm et le modèle une norme finale.

## Pourquoi le contexte et les têtes ne changent pas ce total

`contextLength` change le nombre d'activations et le coût de l'attention. `numHeads` découpe `C` en plusieurs têtes sans changer la taille totale des quatre matrices `[C,C]`. Ces valeurs influencent donc calcul et mémoire, pas le nombre de poids pour l'architecture retenue.

## Preuve actuelle et preuve future

PR01 teste la formule pure. En PR08, nous compterons les paramètres réellement créés par DJL et exigerons l'égalité avec cette estimation.
