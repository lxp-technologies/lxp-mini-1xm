# Objectifs du projet

## Ce que nous construisons

Un Transformer decoder-only autoregressif d'environ 17 millions de paramètres, entraîné from scratch en Kotlin/JVM. Depuis PR05, DJL fournit les tenseurs et l'autograd; l'architecture du modèle reste notre code.

## Pourquoi un petit modèle

Un grand LLM contient les mêmes idées fondamentales, mais sa taille, son infrastructure et ses optimisations les rendent difficiles à observer. Ici, nous privilégions dans cet ordre : correction, compréhension, tests, mesures, puis optimisation.

## Critères de réussite

Nous saurons entraîner notre tokenizer, expliquer chaque forme de tenseur, faire diminuer la loss, restaurer un checkpoint et générer des tokens. La qualité du texte sera une mesure expérimentale, pas une promesse.

## Ce que nous ne construisons pas encore

PR01 ne contient ni tokenizer, ni tenseur, ni attention, ni Transformer, ni boucle d'entraînement. Elle établit seulement une configuration fiable et permet de prévoir la taille du futur modèle.
