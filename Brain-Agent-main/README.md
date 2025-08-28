# Brain AI

Ce projet est un clone de l'interface utilisateur de Claude.ai, créé avec React et Tailwind CSS. Il offre la possibilité d'utiliser différents modèles de langage via leurs APIs.

## Fonctionnalités

- Conversations avec historique persistant (stockage local)
- Support de plusieurs APIs de modèles de langage (OpenAI/GPT, Anthropic/Claude, Google/Gemini)
- Mode sombre / clair
- Suppression des conversations
- Interface responsive (desktop et mobile)
- Animation "thinking" pendant les réponses
- Design modulaire avec composants réutilisables

## Technologies utilisées

- React
- Tailwind CSS
- Context API pour la gestion d'état
- localStorage pour la persistance des données

## Pour démarrer

```bash
# Installation des dépendances
npm install

# Configuration des clés API
cp sample.env .env.local
# Éditez le fichier .env.local pour ajouter vos clés API

# Démarrage du serveur de développement
npm start
```

## Configuration des clés API

Pour utiliser les différentes APIs de modèles de langage, vous devez obtenir les clés API correspondantes:

1. **OpenAI (GPT)**: Obtenez une clé sur [OpenAI Platform](https://platform.openai.com/)
2. **Anthropic (Claude)**: Obtenez une clé sur [Anthropic Console](https://console.anthropic.com/)
3. **Google (Gemini)**: Obtenez une clé sur [Google AI Studio](https://ai.google.dev/)
4. No limit on peut ajouter autant de model qu'on poudrait 

Ajoutez ces clés dans le fichier `.env.local`:

```
REACT_APP_OPENAI_API_KEY=votre-clé-api-openai
REACT_APP_ANTHROPIC_API_KEY=votre-clé-api-anthropic
REACT_APP_GEMINI_API_KEY=votre-clé-api-gemini
```

## Structure du projet

- `components/` : Composants React
  - `chat/` : Composants liés à la zone de chat
  - `layout/` : Composants de structure de page
  - `sidebar/` : Composants pour la barre latérale
  - `ui/` : Composants UI génériques
- `context/` : Context API pour la gestion de l'état
  - `ChatContext.jsx` : Gestion des conversations et modèles
  - `ThemeContext.jsx` : Gestion du thème clair/sombre
- `hooks/` : Hooks personnalisés
- `services/` : Services pour les APIs externes
  - `llmService.js` : Gestion des appels aux APIs des modèles
- `styles/` : Styles globaux (Tailwind CSS)
- `utils/` : Utilitaires et données fictives

## Utilisation

- Sélectionnez un fournisseur et un modèle dans la barre latérale
- Créez une nouvelle conversation avec le bouton "Nouvelle conversation"
- Envoyez des messages et recevez des réponses du modèle sélectionné
- Supprimez des conversations en survolant leur entrée dans la barre latérale
- Basculez entre mode clair et sombre avec le bouton thème

## Personnalisation

- Ajoutez de nouveaux modèles dans `llmService.js`
- Personnalisez les couleurs et styles dans `tailwind.config.js`
- Ajoutez des fonctionnalités additionnelles en étendant les contexts

## Limitations

- Le projet n'inclut pas d'authentification utilisateur
- Les conversations sont stockées localement (localStorage)
- Les APIs de modèles de langage ont des limites d'utilisation
