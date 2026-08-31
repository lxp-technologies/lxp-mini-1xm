# ADR 0012 - Device central et format de chat remplaçable

## Statut

Accepté le 2026-08-31 pour l'extension PR16.

## Décision

- `RuntimeDeviceResolver` est l'unique interprète de `auto`, `cpu` et `cuda:0`.
- L'override CLI précède `runtime.device`, dont le défaut est `auto`.
- La sélection précède le manager racine et devient une métadonnée immuable.
- Une demande CUDA explicite ne retombe jamais silencieusement sur CPU.
- `-PpytorchNative=cpu|cuda` verrouille PyTorch 2.7.1; CUDA utilise cu128.
- Le navigateur possède les tours; le serveur ne garde aucune mémoire conversationnelle.
- `ChatPromptFormatter` isole la sérialisation provisoire et sera remplaçable par le futur chat template SFT.

## Conséquences

Poids, checkpoints, caches et requêtes partagent nécessairement le device du manager racine. Le premier lancement
CUDA télécharge un runtime volumineux. Un GPU n'est pas automatiquement plus rapide : sur de minuscules tensors, le
coût de lancement des kernels peut dominer. Le playground ressemble à un chat, mais sa qualité reste celle du modèle
base.

## Alternatives rejetées

- Détecter CUDA dans chaque commande aurait fragmenté les règles et les tests.
- Replier `cuda:0` vers CPU aurait masqué une mauvaise configuration.
- Installer le CUDA Toolkit complet aurait été inutile pour le runtime PyTorch empaqueté.
- Formater les rôles dans JavaScript aurait couplé l'UI au futur chat template.
- Stocker la conversation côté serveur aurait créé un état implicite hors périmètre.
