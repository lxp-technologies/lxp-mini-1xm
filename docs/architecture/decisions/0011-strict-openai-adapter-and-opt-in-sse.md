# ADR 0011 - Adaptateur OpenAI strict et SSE opt-in

## Statut

Accepté le 2026-08-30 pour PR16.

## Contexte

Le runtime PR14-15 est indépendant du transport et ne supporte qu'un sous-ensemble mesurable des options de
completion. Accepter silencieusement des champs OpenAI donnerait une fausse impression de compatibilité. Le streaming
ajoute aussi une surface opérationnelle : réponse déjà engagée, déconnexion client et ressources natives à fermer.

## Décision

- Spring MVC embarqué adapte HTTP vers `LocalInferenceService`; le coeur `inference` ne dépend pas de Spring.
- Un seul modèle est chargé au démarrage et fermé avec le contexte serveur.
- Le serveur bind `127.0.0.1` par défaut; un bind distant exige `--allow-remote`.
- La désérialisation refuse les propriétés inconnues et chaque option connue mais non implémentée produit
  `unsupported_feature`.
- `REJECT` est la politique de contexte HTTP : aucune troncature implicite du prompt.
- SSE est implémenté dès PR16, mais désactivé par défaut et activé explicitement avec `--streaming-enabled`.
- `stream:false` reste disponible dans les deux configurations.
- Les chunks contiennent les deltas réellement émis pendant la génération, un `finish_reason`, puis `[DONE]`.
- Une erreur d'écriture interrompt la génération et déclenche la fermeture du scope et du cache de requête.

## Conséquences

La compatibilité est petite mais auditable. Un client ne peut pas croire qu'un `stop`, une pénalité ou plusieurs choix
ont été appliqués lorsqu'ils ne le sont pas. Le mode opt-in permet de désactiver la surface SSE sans retirer les
completions JSON.

Une erreur survenant après le premier événement SSE ne peut plus devenir une réponse JSON structurée puisque le statut
HTTP est déjà envoyé; la connexion se ferme alors sans `[DONE]`. Les clients doivent considérer `[DONE]` comme la
preuve d'une terminaison normale.

## Alternatives rejetées

- Ignorer les champs inconnus aurait créé une compatibilité trompeuse.
- Simuler le streaming en découpant un texte déjà généré n'aurait réduit ni la latence du premier token ni la durée de
  vie des ressources après déconnexion.
- Activer SSE sans option aurait élargi silencieusement la surface opérationnelle.
- Placer HTTP dans `InferenceRuntime` aurait couplé le lifecycle DJL à Spring et rendu les tests du coeur plus lourds.
- Ktor était viable, mais Spring MVC correspond à la préférence du projet et fournit le lifecycle embarqué et le type
  `StreamingResponseBody` sans introduire un second modèle de concurrence dans le runtime.
