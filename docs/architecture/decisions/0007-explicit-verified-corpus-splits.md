# ADR 0007 - Fichiers train et validation explicites avec checksums

- Statut : accepté
- Date : 2026-08-30
- Portée : PR12

## Contexte

Évaluer un modèle sur les données qui ont produit ses gradients mesure surtout la mémorisation. Un split après tokenization ne suffit pas si le tokenizer BPE a lui-même appris ses merges sur le fichier complet : la validation a alors influencé la représentation.

PR12 doit aussi comparer plusieurs checkpoints sur exactement les mêmes tokens et refuser une association accidentelle avec un autre corpus ou tokenizer.

## Décision

`train corpus` exige deux fichiers UTF-8 distincts : `--train-corpus` et `--validation-corpus`.

- Le BPE est entraîné séparément sur train seulement avant la commande.
- Chaque fichier possède son propre lecteur streaming et ses propres fenêtres.
- Le run enregistre séparément SHA-256 train, validation, tokenizer et configuration.
- Le tokenizer est copié dans le run.
- `experiment.json` fixe les options hors YAML.
- `evaluate` recharge strictement les métadonnées et refuse un checksum différent.
- L'évaluation utilise `training=false`, aucun `GradientCollector` et une moyenne pondérée par token.
- Les checkpoints sont comparés par identifiant contrôlé dans un seul run.

## Pourquoi

Deux fichiers rendent la frontière visible, auditable et difficile à franchir par erreur. Les checksums transforment « probablement les mêmes données » en invariant vérifiable. Un service d'évaluation distinct permet de tester l'absence de mutation des poids indépendamment de la CLI.

## Conséquences

- L'utilisateur prépare explicitement train et validation avant le run.
- Un tokenizer appris sur le fichier complet n'est pas détectable par magie; sa procédure et son checksum doivent être consignés. Le laboratoire montre la commande correcte.
- Modifier un seul byte de validation bloque `evaluate`.
- Les derniers fragments trop courts pour `contextLength + 1` ne participent pas à la métrique.
- Le débit cumulatif du run inclut l'observation; le débit `evaluate` ne couvre que le forward validation.
- Les samples invalides UTF-8 sont conservés sous forme d'IDs et ne stoppent pas l'entraînement.
- La reprise exacte reste absente : comparer des checkpoints d'un run frais n'implique pas que ses moments AdamW soient restaurables.

## Alternatives écartées

- **Split unique après BPE appris sur tout le corpus :** fuite de validation dans les merges.
- **Faire confiance aux chemins sans hash :** un fichier peut changer en gardant le même nom.
- **Moyenne non pondérée des batches :** biais lorsque le dernier batch est plus petit.
- **Évaluer avec backward puis jeter les gradients :** travail inutile et risque de mutation.
- **Choisir le meilleur checkpoint sur train loss :** ne détecte pas le surapprentissage.
- **Lancer immédiatement 17 M dès que la validation baisse :** ignore qualité qualitative, échelle des données et profilage.

## Preuves exigées

- test de readers train/validation physiquement séparés;
- test de perplexité connue;
- poids et gradients inchangés après évaluation;
- run CLI produisant métriques, samples et plusieurs checkpoints;
- rechargement de deux checkpoints sur la même validation;
- corpus validation modifié refusé par checksum;
- expérience réelle avec conclusion explicite sur la porte 17 M.
