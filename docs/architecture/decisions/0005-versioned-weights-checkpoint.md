# ADR 0005 - Checkpoint de poids versionné sans fausse reprise exacte

- Statut : accepté
- Date : 2026-08-30
- Portée : PR10

## Contexte

Un fichier de poids isolé ne suffit pas à reproduire un run ni à reprendre AdamW exactement. Le plan exige sauvegarde, checksums, métadonnées et une preuve de continuité de l'optimizer ou une limitation déclarée.

DJL 0.36 fournit `Block.saveParameters/loadParameters` et `Optimizer.Builder.optBeginNumUpdate`. Son AdamW conserve toutefois les moments par paramètre dans des champs privés et ne fournit aucune API publique de sérialisation de cet état.

## Décision

Nous définissons un format de run et de checkpoint versionné par l'application :

- paramètres binaires DJL, sans inventer notre propre encodage tensoriel;
- manifeste JSON strict avec SHA-256 du modèle et de la configuration;
- métadonnées du run et métriques JSONL;
- progression restaurée à une frontière d'update;
- compteur AdamW et scheduler repositionnés;
- `optimizerMomentsRestored=false`, `randomStateRestored=false` et `exactTrainingResume=false` obligatoires dans le format 1.

Le manifeste est écrit après le modèle et `latest.txt` après le manifeste. Un dossier existant n'est jamais écrasé.

## Pourquoi

Cette solution apporte une restauration exacte des poids et des logits avec un format inspectable. Elle protège contre corruption et mauvaise configuration, tout en refusant de présenter une continuation approximative comme identique au run original.

Réimplémenter AdamW uniquement pour sérialiser son état augmenterait fortement le risque numérique et contredirait la décision pédagogique de PR09. Lire ses champs privés par réflexion rendrait le checkpoint dépendant d'une implémentation non contractuelle de DJL.

## Conséquences

- Un checkpoint PR10 convient à l'inférence future et à une continuation exploratoire.
- Une continuation après interruption peut suivre une trajectoire différente, même si update, tokens et learning rate reprennent correctement.
- Le format doit évoluer avant toute affirmation de reprise exacte.
- Les checkpoints doivent être pris après une update, jamais au milieu d'une accumulation.
- Le tokenizer est absent du run synthétique; il deviendra obligatoire pour les vrais runs.
- Une corruption des poids ou un changement de configuration provoque une erreur avant chargement.

## Alternatives écartées

- **Sauver seulement `model.params` :** aucune provenance, compatibilité ou progression vérifiable.
- **Réflexion sur les champs privés AdamW :** fragile entre versions et non supportée par DJL.
- **Déclarer la reprise exacte sans moments :** assertion fausse; la prochaine update dépend de `m` et `v`.
- **Écraser `latest` en place avant les poids :** risque de pointer vers un checkpoint incomplet.
- **Sérialisation Java native :** format peu stable, opaque et inadapté aux ressources natives DJL.

## Preuves exigées

- nouveau modèle et nouveau manager après sauvegarde;
- logits exactement identiques après chargement;
- backward fonctionnel sur les paramètres restaurés;
- update, tokens, scheduler et compteur poursuivis;
- checksum corrompu et configuration différente refusés;
- manifeste incapable d'affirmer une reprise exacte en format 1;
- CLI affichant explicitement tous les états non restaurés.
