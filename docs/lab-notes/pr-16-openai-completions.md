# PR16 - Servir une completion JSON ou SSE

## Question

Peut-on exposer le même runtime chargé une seule fois avec un contrat OpenAI-compatible strict, puis choisir au
démarrage si les clients ont le droit d'utiliser le streaming?

## Préparer un run neuf

Depuis PowerShell à la racine :

```powershell
$labId = Get-Date -Format yyyyMMdd-HHmmss
$runDir = "build/labs/pr16/demo-$labId"
$tokenizerPath = "build/labs/pr16/tokenizer-$labId.json"
.\gradlew.bat run --args="tokenizer byte create --output $tokenizerPath"
.\gradlew.bat run --args="train checkpoint-demo --config configs/lab-pr09-tiny.yaml --run-dir $runDir --before-updates 80 --after-updates 1"
```

Le suffixe rend le dossier neuf. `checkpoint-demo` refuse volontairement un dossier existant ou non vide.

## Expérience A - streaming activé

Dans ce terminal, conserver `$runDir` et `$tokenizerPath` :

```powershell
.\gradlew.bat run --args="serve --model-id lxp-mini-pr16-tiny-base --run-dir $runDir --tokenizer $tokenizerPath --port 8080 --streaming-enabled"
```

Dans un second PowerShell :

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health

$normalBody = @{
    model = "lxp-mini-pr16-tiny-base"
    prompt = "abc"
    max_tokens = 5
    temperature = 0
    stream = $false
} | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/completions `
    -ContentType application/json -Body $normalBody | ConvertTo-Json -Depth 8

$streamBody = @{
    model = "lxp-mini-pr16-tiny-base"
    prompt = "abc"
    max_tokens = 5
    temperature = 0
    stream = $true
} | ConvertTo-Json -Compress
$streamBody | curl.exe -sN -H "Content-Type: application/json" --data-binary '@-' `
    http://127.0.0.1:8080/v1/completions
```

`curl.exe -N` évite de bufferiser l'affichage. Avec le tiny model de référence, le JSON normal a produit `" abc "` et
le SSE a produit les deltas `" "`, `"a"`, `"b"`, `"c"`, `" "`, suivis de `finish_reason:"length"` et `[DONE]`.
Les deux textes sont identiques.

## Expérience B - streaming désactivé

Arrêter le premier serveur avec `Ctrl+C`, puis :

```powershell
.\gradlew.bat run --args="serve --model-id lxp-mini-pr16-tiny-base --run-dir $runDir --tokenizer $tokenizerPath --port 8080 --no-streaming-enabled"
```

Renvoyer `$normalBody` fonctionne toujours. Renvoyer `$streamBody` doit retourner HTTP `400` avec
`code:"unsupported_feature"` et `param:"stream"`.

## Piège observé : contexte 8

La configuration `lab-pr09-tiny.yaml` possède un contexte de 8 tokens. Le prompt byte `abc` en consomme 3; il ne
reste donc que 5 tokens. Une requête `max_tokens:12` échoue correctement avec `context_length_exceeded` au lieu de
tronquer silencieusement le prompt.

## Tests automatisés

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.server.InferenceHttpServerTest"
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.inference.InferenceRuntimeTest"
.\gradlew.bat test
```

Le contract test démarre un vrai serveur sur un port libre, compare API et runtime direct, vérifie modèles, usage,
champs inconnus, fonctions non supportées, contexte trop long, SSE et refus lorsque désactivé. Le test runtime simule
une déconnexion en faisant échouer le callback et vérifie que le nombre d'arrays managées reste stable.

## Résultat mesuré

Le 2026-08-30 sur Windows/JDK 25, le laboratoire a produit un checkpoint `step-00000081`. `/health` annonçait
`streaming_enabled:true`; la réponse normale comptait `3` tokens de prompt, `5` de completion et `8` au total. La
suite complète terminait par `BUILD SUCCESSFUL`.

## Conclusion

La porte PR16 est franchie : lifecycle serveur, contrat strict, comparaison runtime/API, usage et streaming réel sont
testés. PR17 peut désormais se concentrer sur le durcissement du streaming, notamment un soak test de centaines de
requêtes et le comportement de clients réels, plutôt que sur une première implémentation SSE.

## Extension device - environnement observé

Mesure du 2026-08-31 :

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 188 MiB |
| driver | 591.74 |
| CUDA maximal vu par le driver | 13.1 |
| DJL / PyTorch | 0.36.0 / 2.7.1 |
| runtime CUDA | cu128 |

`nvidia-smi` affichait environ `573 MiB` de VRAM déjà utilisée par le bureau avant le run; cette mémoire
n'appartenait pas au modèle.

```powershell
.\gradlew.bat run --args="runtime info --device auto"
.\gradlew.bat -PpytorchNative=cpu run --args="runtime info --device cpu"
.\gradlew.bat -PpytorchNative=cuda run --args="runtime info --device cuda:0"
nvidia-smi
```

Sans flavor CUDA chargé, `auto` a choisi CPU et `cuda:0` a échoué explicitement. Avec
`-PpytorchNative=cuda`, DJL a téléchargé PyTorch/cu128, détecté un GPU et sélectionné `cuda:0`.

### Même checkpoint et même seed

CPU et GPU ont utilisé `step-00000081`, `prompt=abc`, greedy, seed 42 et cinq nouveaux tokens. Les deux ont produit
exactement `" abc "`. Le débit observé était `122,74 tokens/s` sur CPU et `16,35 tokens/s` sur GPU. Ce modèle de
`6 752` paramètres est trop petit pour amortir le lancement des kernels CUDA.

### Benchmark contexte 256

| Device | Tokens | cache tokens/s | recalcul tokens/s | sorties identiques |
|---|---:|---:|---:|---|
| CPU | 32 | 763,53 | 1 029,75 | oui |
| CPU | 64 | 1 110,60 | 968,29 | oui |
| CPU | 128 | 1 708,77 | 1 106,10 | oui |
| CUDA | 32 | 352,21 | 442,38 | oui |
| CUDA | 64 | 421,97 | 543,56 | oui |
| CUDA | 128 | 598,00 | 567,86 | oui |

Cette mesure prouve la propagation et l'équivalence sur le tiny laboratoire; elle ne permet pas de conclure que le
CPU restera plus rapide sur le modèle 17 M.

### Observer le playground

Préparer un run avec `configs/lab-pr15-kv-cache.yaml`, démarrer `serve`, puis ouvrir
`http://localhost:8080/`. Comparer une continuation avec et sans préfixe system. Un modèle base peut répéter ou
continuer les libellés : il n'obéit pas encore à une hiérarchie de rôles. Le bouton Clear vide l'écran sans requête
serveur.

Le smoke test réel a chargé le modèle CPU à contexte 256, servi `/` et `/app.css` avec HTTP 200, puis formaté
`System: Be concise.\nUser: abc\nAssistant:` en 40 tokens. Sa continuation greedy de huit tokens était `EEEEEEEE` :
le transport et le formatage fonctionnent, mais la sortie illustre précisément pourquoi ce checkpoint base n'est pas
encore un chatbot.

Une vérification supplémentaire avec `Hi!` a confirmé la frontière : le checkpoint quick-start de `6 752`
paramètres produit des bytes Unicode valides, mais pas une phrase anglaise, puisqu'il a seulement vu `abc `. Les runs
TinyStories 17 M présents au moment de l'expérience n'avaient que deux updates et une validation loss proche de
`6,19`; ils ne constituent pas encore un checkpoint linguistique. Aucun filtre de caractères ne peut remplacer les
données et les optimizer updates manquants.

## Expérience D - Reproduire une sortie UTF-8 invalide

Le tiny modèle peut favoriser des byte tokens qui ne forment pas du UTF-8 valide. Avant le correctif PR16, le
playground retournait une erreur 500; le premier correctif remplaçait ces bytes par `�`. La contrainte finale les
retire maintenant de la distribution avant le sampling. Ces tests couvrent la séquence observée, l'automate UTF-8,
le masque du sampler et l'équivalence streaming :

```powershell
.\gradlew.bat test --tests "io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizerTest" --tests "io.github.lxptechnologies.lxpmini.inference.Utf8TokenConstraintTest" --tests "io.github.lxptechnologies.lxpmini.inference.IncrementalTextDecoderTest" --tests "io.github.lxptechnologies.lxpmini.generation.TokenSamplerTest"
```

Résultat attendu : `BUILD SUCCESSFUL`. Pour vérifier manuellement dans le playground, arrêter l'ancien serveur avec
`Ctrl+C`, relancer la même commande `serve`, puis générer avec le même prompt et la même seed. La réponse ne doit
contenir ni erreur 500 ni `�`. Elle peut rester incohérente ou contenir des caractères Unicode inattendus : la
contrainte garantit l'encodage, tandis que la qualité linguistique dépend de l'entraînement du modèle.
