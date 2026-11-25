# Suggestion de tags Stack Overflow

Suggérer automatiquement des tags pour une question Stack Overflow, à partir de son titre.

## Language

**Post**:
Une question Stack Overflow, identifiée par son `post_id`, avec un titre et de 1 à 5 tags.
_Avoid_: question, ligne, exemple

**Titre**:
Le texte court (environ 8 mots) d'un post. C'est la seule entrée disponible pour la suggestion.
_Avoid_: question, texte, body

**Tag**:
Un libellé Stack Overflow attaché à un post (ex. `c#`, `asp.net-mvc`, `null`). Un post en porte plusieurs, sans hiérarchie entre eux.
_Avoid_: label, catégorie, classe

**Tag principal**:
Le tag en `tag_position = 0` d'un post. Il n'est pas utilisé comme cible.

**Vocabulaire de tags**:
L'ensemble des K tags les plus fréquents (K = 50 par défaut) qu'un modèle sait suggérer. Les autres tags sont hors vocabulaire.
_Avoid_: classes, label set

**Suggestion**:
La liste classée des k tags du vocabulaire jugés les plus probables pour un titre, chacun avec son score (k = 3 par défaut).
_Avoid_: prédiction, recommandation

**Modèle classique**:
Le suggesteur de référence, fondé sur des représentations de mots sans pré-entraînement.
_Avoid_: baseline (comme nom propre), modèle simple

**Modèle BERT**:
Le suggesteur fondé sur un BERT pré-entraîné, fine-tuné sur les posts.
_Avoid_: modèle avancé, transformer
