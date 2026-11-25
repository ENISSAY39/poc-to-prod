# Modèle classique : TF-IDF mots + n-grammes de caractères et régression logistique one-vs-rest

Le modèle classique suggère des tags à partir du titre seul. Les titres sont très courts (environ 8 mots) et pleins de tokens techniques (`c#`, `.net`, `asp.net-mvc`). Les 50 tags du vocabulaire ont chacun au moins 33 exemples, sur environ 3 800 posts. On utilise donc un TF-IDF sur les mots, plus des n-grammes de caractères, suivi d'une régression logistique par tag (one-vs-rest). Les tags sont triés par probabilité pour produire le top-3. `C` est choisi dans {0.1, 1, 10} sur le micro-F1 de validation.

## Considered Options

- **Naive Bayes multinomial** : rapide, mais ses probabilités sont mal calibrées, alors que la suggestion trie les tags par score.
- **Embeddings moyennés (word2vec / fastText)** : les embeddings pré-entraînés couvrent mal les tokens de code. Les entraîner sur 4 600 titres ne suffit pas.
- **SVM linéaire** : précision comparable, mais il ne donne pas de probabilités directement.

## Consequences

Il n'y a pas de stemming ni de suppression des caractères spéciaux : les n-grammes de caractères doivent garder `#`, `+` et `.`, qui distinguent `c`, `c#` et `c++`.
