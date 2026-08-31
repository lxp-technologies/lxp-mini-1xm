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
