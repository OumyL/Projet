/**
 * Service pour gérer les connexions aux différentes APIs de modèles de langage
 * via notre serveur proxy pour éviter les problèmes CORS
 * MODIFIÉ : Intégration avec MCP-Trader
 */

const PROXY_URL = 'http://localhost:3001';

// Configuration des clés d'API et des points d'accès
const API_CONFIG = {
  OPENAI: {
    apiKey: process.env.REACT_APP_OPENAI_API_KEY,
    endpoint: `${PROXY_URL}/api/openai`,
    models: ['gpt-4o-mini', 'gpt-3.5-turbo'],
    priority: 2
    
  },
  ANTHROPIC: {
    apiKey: process.env.REACT_APP_OPENAI_API_KEY,
    endpoint: `${PROXY_URL}/api/anthropic`,
    models: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
    priority: 1,
    hasCredits: false
  },
  GEMINI: {
    apiKey: '',
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models',
    models: ['gemini-pro', 'gemini-ultra'],
    priority: 3
  }
};
// Validation des clés API au démarrage
const validateAPIKeys = () => {
  const missingKeys = [];
  
  if (!API_CONFIG.OPENAI.apiKey) {
    missingKeys.push('REACT_APP_OPENAI_API_KEY');
  }
  
  if (!API_CONFIG.ANTHROPIC.apiKey) {
    missingKeys.push('REACT_APP_ANTHROPIC_API_KEY');
  }
  
  if (missingKeys.length > 0) {
    console.error('Clés API manquantes:', missingKeys);
    console.error('Vérifiez votre fichier .env.local');
  }
  
  return missingKeys.length === 0;
};

// Appelez la validation au chargement du module
validateAPIKeys();
//Système de fallback intelligent
const FALLBACK_ORDER = ['OPENAI', 'ANTHROPIC', 'GEMINI'].filter(provider => 
  API_CONFIG[provider].apiKey && API_CONFIG[provider].apiKey.length > 10
);

// FONCTIONS MCP
// Détecter les requêtes de trading
const isCreditsError = (error) => {
  const errorStr = JSON.stringify(error).toLowerCase();
  return errorStr.includes('credit') || 
         errorStr.includes('balance') || 
         errorStr.includes('billing') ||
         errorStr.includes('quota');
};

// Détecter les erreurs de rate limit
const isRateLimitError = (error) => {
  const errorStr = JSON.stringify(error).toLowerCase();
  return errorStr.includes('429') || 
         errorStr.includes('rate') || 
         errorStr.includes('limit');
};

// AMÉLIORATION : Appel avec fallback automatique
const callLLMWithAutoFallback = async (message, preferredProvider = null, preferredModel = null) => {
  console.log(`🧠 BRAIN: Starting smart LLM call with fallback...`);
  
  // Ordonner les providers selon préférence
  let providersToTry = [...FALLBACK_ORDER];
  if (preferredProvider && FALLBACK_ORDER.includes(preferredProvider)) {
    // Mettre le provider préféré en premier
    providersToTry = [preferredProvider, ...FALLBACK_ORDER.filter(p => p !== preferredProvider)];
  }

  const errors = [];
  
  for (let i = 0; i < providersToTry.length; i++) {
    const provider = providersToTry[i];
    const config = API_CONFIG[provider];
    
    // Skip si pas de crédit détecté précédemment
    if (config.hasCredits === false && provider === 'ANTHROPIC') {
      console.log(`⚠️ Skipping ${provider} - no credits detected`);
      continue;
    }

    try {
      console.log(`🔄 Trying ${provider} (attempt ${i + 1}/${providersToTry.length})`);
      
      let model = preferredModel;
      if (!model || !config.models.includes(model)) {
        model = config.models[0]; // Prendre le premier modèle disponible
      }

      const response = await callSelectedLLM(message, provider, model);
      
      // Marquer comme fonctionnel
      if (provider === 'ANTHROPIC') {
        config.hasCredits = true;
      }
      
      console.log(`✅ Success with ${provider}/${model}`);
      return {
        response,
        provider,
        model,
        fallbackUsed: i > 0
      };

    } catch (error) {
      console.error(`❌ ${provider} failed:`, error.message);
      errors.push({ provider, error: error.message });
      
      // Marquer Anthropic comme sans crédit si erreur de facturation
      if (provider === 'ANTHROPIC' && isCreditsError(error)) {
        console.log(`💳 Marking ${provider} as no credits`);
        config.hasCredits = false;
      }
      
      // Attendre un peu avant le prochain essai si rate limit
      if (isRateLimitError(error) && i < providersToTry.length - 1) {
        console.log(`⏳ Rate limit detected, waiting 2s before next provider...`);
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      
      continue;
    }
  }

  // Tous les providers ont échoué
  const errorSummary = errors.map(e => `${e.provider}: ${e.error}`).join('\n');
  throw new Error(`All LLM providers failed:\n${errorSummary}`);
};
const isTradingQuery = (message) => {
  const tradingKeywords = [
    'analys', 'analyse', 'action', 'stock', 'crypto', 'bitcoin', 'ethereum',
    'recommand', 'conseil', 'investir', 'acheter', 'vendre', 'marché', 'trading',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC', 'ETH'
  ];
  return tradingKeywords.some(keyword => 
    message.toLowerCase().includes(keyword.toLowerCase())
  );
};

// Extraire symboles boursiers
const extractSymbols = (message) => {
  const symbols = message.match(/\b[A-Z]{1,5}\b/g) || [];
  return symbols.filter(s => s.length <= 5);
};

// Appeler MCP pour analyse
const callMCPAnalysis = async (message) => {
  const symbols = extractSymbols(message);
  if (symbols.length === 0) return null;
  
  try {
    console.log(`📊 Appel MCP pour ${symbols[0]}`);
    const response = await fetch('http://localhost:3001/api/mcp/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: symbols[0] })
    });
    
    const data = await response.json();
    if (data.success) {
      return `📊 **Analyse de ${data.symbol}**\n\n${data.data}`;
    } else {
      console.error('Erreur MCP:', data.error);
      return null;
    }
  } catch (error) {
    console.error('Erreur MCP:', error);
    return null;
  }
};

/**
 * Envoie une requête à l'API OpenAI via notre proxy
 */
const callOpenAI = async (message, model = 'gpt-4o-mini') => {
  try {
    console.log(`Calling OpenAI API with model ${model}...`);
    
    const response = await fetch(API_CONFIG.OPENAI.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_CONFIG.OPENAI.apiKey}`
      },
      body: JSON.stringify({
        model: model,
        messages: [{ role: 'user', content: message }],
        temperature: 0.7,
        max_tokens: 2000
      })
    });

     const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error?.message || `OpenAI API error: ${response.status}`);
    }
    
    return data.choices[0].message.content;
  } catch (error) {
    console.error('Erreur OpenAI:', error);
    throw error;
  }
};


/**
 * Envoie une requête à l'API Anthropic (Claude) via notre proxy
 */
const callAnthropic = async (message, model = 'claude-3-haiku-20240307') => {
  try {
    console.log(`Calling Anthropic API with model ${model}...`);
    
    const requestBody = {
      model: model,
      messages: [{ role: 'user', content: message }],
      max_tokens: 1000,
      temperature: 0.7,
      system: "Vous êtes BRAIN, un assistant IA spécialisé en finance et trading, utile et professionnel."
    };
    
    const response = await fetch(API_CONFIG.ANTHROPIC.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': API_CONFIG.ANTHROPIC.apiKey
      },
      body: JSON.stringify(requestBody)
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error?.message || `Erreur API Anthropic: ${JSON.stringify(data.error)}`);
    }
    
    const textContent = data.content.find(item => item.type === 'text');
    if (!textContent || !textContent.text) {
      throw new Error('Pas de contenu textuel dans la réponse');
    }
    
    return textContent.text;
  } catch (error) {
    console.error('Erreur Anthropic:', error);
    throw error;
  }
};

/**
 * Envoie une requête à l'API Gemini de Google
 */
const callGemini = async (message, model = 'gemini-pro') => {
  try {
    const response = await fetch(`${API_CONFIG.GEMINI.endpoint}/${model}:generateContent?key=${API_CONFIG.GEMINI.apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        contents: [{ parts: [{ text: message }] }],
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 1000
        }
      })
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error?.message || 'Erreur lors de l\'appel à l\'API Gemini');
    }
    
    return data.candidates[0].content.parts[0].text;
  } catch (error) {
    console.error('Erreur Gemini:', error);
    throw error;
  }
};

/**
 * Fonction helper pour appeler le LLM sélectionné
 */
const callSelectedLLM = async (message, provider, model) => {
  // Vérifier le serveur proxy
  if (provider === 'OPENAI' || provider === 'ANTHROPIC') {
    const isProxyRunning = await checkProxyServer().catch(() => false);
    if (!isProxyRunning) {
      throw new Error('Le serveur proxy n\'est pas en cours d\'exécution.');
    }
  }
  
  switch (provider) {
    case 'OPENAI':
      return await callOpenAI(message, model);
    case 'ANTHROPIC':
      return await callAnthropic(message, model);
    case 'GEMINI':
      return await callGemini(message, model);
    default:
      throw new Error('Fournisseur de modèle non pris en charge');
  }
};

/**
 * Vérifie si le serveur proxy est en cours d'exécution
 */
export const checkProxyServer = async () => {
  try {
    const response = await fetch(`${PROXY_URL}/api/health`);
    const data = await response.json();
    return data.status === 'ok';
  } catch (error) {
    console.error('Erreur lors de la vérification du serveur proxy:', error);
    return false;
  }
};

// Mode démo MCP (quand l'API ne répond pas)
const getMockMCPAnalysis = (symbol) => {
  return `📊 **Analyse technique de ${symbol}** (Mode démo MCP-Trader)

**Tendance générale** : Haussière à court terme
**RSI (14)** : 65.4 (Zone neutre-surachat) 
**MACD** : Signal d'achat récent
**Bollinger Bands** : Prix proche de la bande supérieure
**Volume** : Supérieur à la moyenne mobile 20 jours

**Support/Résistance** :
- Support clé : $150.00
- Résistance : $185.00
- Prix actuel : ~$175.00

**Recommandation** : Position modérément haussière avec prise de profit partielle si dépassement de $180.

*Analyse générée par MCP-Trader - Version démo*`;
};

/**
 * FONCTION PRINCIPALE MODIFIÉE
 * Gère les requêtes de trading avec MCP et les requêtes normales avec LLM
 */
export const generateResponse = async (message, provider, model) => {
  console.log(`🧠 BRAIN: Traitement du message avec ${provider}/${model}`);
  
  // 1. Vérifier si c'est une requête de trading
  if (isTradingQuery(message)) {
    console.log('📊 Requête de trading détectée');
    
    const mcpResult = await callMCPAnalysis(message);
    if (mcpResult) {
      // Enrichir avec le LLM pour une meilleure présentation
      const enrichedPrompt = `Tu es BRAIN, un assistant de trading professionnel. 

Analyse MCP-Trader :
${mcpResult}

INSTRUCTIONS SPÉCIALES POUR GEMINI :
- Utilise un format markdown structuré
- Ajoute des sections claires (## Titre)
- Inclus des conseils pratiques
- Sois professionnel mais accessible
- Évite les émojis excessifs`;
      
      try {
        console.log('✨ Enrichissement avec le LLM');
        const llmResponse = await callSelectedLLM(enrichedPrompt, provider, model);
        return llmResponse;
      } catch (error) {
        console.log('⚠️ Échec enrichissement LLM, retour analyse brute');
        return mcpResult;
      }
    } else {
      console.log('❌ Aucune donnée MCP, traitement normal');
    }
  }
  
  // 2. Traitement normal avec LLM
  console.log('💭 Traitement normal avec LLM');
  return await callSelectedLLM(message, provider, model);
};

/**
 * Récupère la liste de tous les modèles disponibles
 */
export const getAvailableModels = () => {
  return {
    OPENAI: API_CONFIG.OPENAI.models,
    ANTHROPIC: API_CONFIG.ANTHROPIC.models,
    GEMINI: API_CONFIG.GEMINI.models
  };
};

/**
 * Vérifie si la clé API pour un fournisseur donné est configurée
 */
export const isProviderConfigured = (provider) => {
  return !!API_CONFIG[provider].apiKey;
};