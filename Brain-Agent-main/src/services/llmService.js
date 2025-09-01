/**
 * Service LLM sécurisé pour BRAIN Trading Assistant
 * Communique avec le backend sécurisé - AUCUNE CLÉ API EXPOSÉE
 */

// Configuration - Seulement des URLs publiques
const CONFIG = {
  API_URL: process.env.REACT_APP_API_URL || 'http://localhost:3001',
  TIMEOUT: 30000,
  MAX_RETRIES: 2,
  CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
  MAX_CACHE_SIZE: 100
};

// Cache simple pour les réponses
class ResponseCache {
  constructor() {
    this.cache = new Map();
  }

  // Générer une clé de cache basée sur le message
  generateKey(message, provider, model) {
    const hash = this.simpleHash(message + provider + model);
    return `${provider}:${model}:${hash}`;
  }

  // Hash simple pour la clé de cache
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convertir en entier 32 bits
    }
    return Math.abs(hash).toString(36);
  }

  // Récupérer du cache
  get(key) {
    const item = this.cache.get(key);
    if (!item) return null;

    // Vérifier l'expiration
    if (Date.now() - item.timestamp > CONFIG.CACHE_DURATION) {
      this.cache.delete(key);
      return null;
    }

    return item.data;
  }

  // Stocker en cache avec nettoyage automatique
  set(key, data) {
    // Nettoyer le cache si trop plein
    if (this.cache.size >= CONFIG.MAX_CACHE_SIZE) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  // Statistiques du cache
  getStats() {
    return {
      size: this.cache.size,
      maxSize: CONFIG.MAX_CACHE_SIZE,
      hitRate: this.hitCount / (this.hitCount + this.missCount) * 100 || 0
    };
  }

  clear() {
    this.cache.clear();
    this.hitCount = 0;
    this.missCount = 0;
  }
}

// Instance globale du cache
const cache = new ResponseCache();

// Classe d'erreur personnalisée
class LLMError extends Error {
  constructor(message, type, status, provider) {
    super(message);
    this.name = 'LLMError';
    this.type = type;
    this.status = status;
    this.provider = provider;
    this.timestamp = new Date().toISOString();
  }
}

// Service principal
class LLMService {
  constructor() {
    this.requestCount = 0;
    this.errorCount = 0;
    this.lastErrorTime = null;
  }

  /**
   * Méthode principale pour générer des réponses
   * @param {string} message - Le message à envoyer
   * @param {object} options - Options de configuration
   * @returns {Promise<object>} - Réponse du LLM
   */
  async generateResponse(message, options = {}) {
    const {
      provider = 'openai',
      model,
      temperature = 0.7,
      maxTokens,
      useCache = true
    } = options;

    // Validation des entrées
    this.validateInput(message, provider);

    // Modèle par défaut selon le provider
    const selectedModel = model || (provider === 'openai' ? 'gpt-4o-mini' : 'claude-3-haiku-20240307');

    // Vérifier le cache d'abord
    if (useCache) {
      const cacheKey = cache.generateKey(message, provider, selectedModel);
      const cachedResponse = cache.get(cacheKey);
      
      if (cachedResponse) {
        console.log('[Cache] Réponse trouvée en cache');
        return {
          ...cachedResponse,
          fromCache: true,
          cacheKey
        };
      }
    }

    // Préparer les données de la requête
    const requestData = {
      messages: [{ role: 'user', content: message }],
      model: selectedModel,
      temperature,
      ...(maxTokens && { max_tokens: maxTokens })
    };

    try {
      this.requestCount++;
      console.log(`[${provider}] Envoi requête - ${selectedModel}`);

      const response = await this.makeSecureRequest(provider, requestData);
      
      const result = {
        content: this.extractContent(response, provider),
        provider,
        model: selectedModel,
        usage: response.usage || null,
        timestamp: new Date().toISOString(),
        fromCache: false
      };

      // Mettre en cache si demandé et pas d'erreur
      if (useCache && result.content) {
        const cacheKey = cache.generateKey(message, provider, selectedModel);
        cache.set(cacheKey, result);
      }

      console.log(`[${provider}] Succès - ${result.content.length} caractères`);
      return result;

    } catch (error) {
      this.errorCount++;
      this.lastErrorTime = new Date().toISOString();
      
      console.error(`[${provider}] Erreur:`, error.message);

      // Tentative de fallback automatique
      if (provider === 'openai' && options.allowFallback !== false) {
        console.log('[Fallback] Tentative avec Anthropic...');
        try {
          return await this.generateResponse(message, {
            ...options,
            provider: 'anthropic',
            allowFallback: false // Éviter la boucle infinie
          });
        } catch (fallbackError) {
          console.error('[Fallback] Échec:', fallbackError.message);
          // Lancer l'erreur originale, pas celle du fallback
        }
      }

      throw error;
    }
  }

  /**
   * Faire une requête sécurisée au backend
   */
  async makeSecureRequest(provider, data, retryCount = 0) {
    const url = `${CONFIG.API_URL}/api/${provider}`;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.TIMEOUT);

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new LLMError(
          errorData.error?.message || `Erreur HTTP ${response.status}`,
          errorData.error?.type || 'api_error',
          response.status,
          provider
        );
      }

      return await response.json();

    } catch (error) {
      clearTimeout(timeoutId);

      // Gestion des différents types d'erreurs
      if (error.name === 'AbortError') {
        throw new LLMError('Timeout de la requête', 'timeout_error', 408, provider);
      }

      if (error instanceof LLMError) {
        // Retry automatique pour certaines erreurs
        if (this.shouldRetry(error) && retryCount < CONFIG.MAX_RETRIES) {
          const delay = Math.pow(2, retryCount) * 1000; // Backoff exponentiel
          console.log(`[${provider}] Retry ${retryCount + 1} dans ${delay}ms...`);
          
          await this.sleep(delay);
          return this.makeSecureRequest(provider, data, retryCount + 1);
        }
        throw error;
      }

      // Erreur réseau générique
      throw new LLMError('Erreur de connexion au serveur', 'network_error', 0, provider);
    }
  }

  /**
   * Extraire le contenu de la réponse selon le provider
   */
  extractContent(response, provider) {
    switch (provider) {
      case 'openai':
        return response.choices?.[0]?.message?.content || '';
        
      case 'anthropic':
        const textContent = response.content?.find(item => item.type === 'text');
        return textContent?.text || '';
        
      default:
        return response.text || JSON.stringify(response);
    }
  }

  /**
   * Valider les entrées
   */
  validateInput(message, provider) {
    if (!message || typeof message !== 'string') {
      throw new LLMError('Message requis (string)', 'validation_error', 400);
    }

    if (message.length === 0) {
      throw new LLMError('Message ne peut pas être vide', 'validation_error', 400);
    }

    if (message.length > 10000) {
      throw new LLMError('Message trop long (max 10000 caractères)', 'validation_error', 400);
    }

    if (!['openai', 'anthropic'].includes(provider)) {
      throw new LLMError('Provider non supporté', 'validation_error', 400);
    }
  }

  /**
   * Déterminer si on doit retry une requête
   */
  shouldRetry(error) {
    // Pas de retry pour les erreurs d'authentification ou de validation
    if (error.status === 401 || error.status === 400) return false;
    
    // Retry pour les erreurs de rate limit et serveur
    if (error.status === 429 || error.status >= 500) return true;
    
    // Retry pour les erreurs réseau
    if (error.type === 'network_error' || error.type === 'timeout_error') return true;
    
    return false;
  }

  /**
   * Utilitaire pour attendre
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Méthodes de convenance pour les providers spécifiques
   */
  async askOpenAI(message, model = 'gpt-4o-mini', options = {}) {
    return this.generateResponse(message, { ...options, provider: 'openai', model });
  }

  async askAnthropic(message, model = 'claude-3-haiku-20240307', options = {}) {
    return this.generateResponse(message, { ...options, provider: 'anthropic', model });
  }

  /**
   * Vérifier la santé du service backend
   */
  async checkHealth() {
    try {
      const response = await fetch(`${CONFIG.API_URL}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // Timeout court pour health check
      });

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      const data = await response.json();
      return {
        healthy: true,
        ...data,
        frontend: {
          cacheStats: cache.getStats(),
          requestCount: this.requestCount,
          errorCount: this.errorCount,
          lastErrorTime: this.lastErrorTime
        }
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message,
        frontend: {
          cacheStats: cache.getStats(),
          requestCount: this.requestCount,
          errorCount: this.errorCount,
          lastErrorTime: this.lastErrorTime
        }
      };
    }
  }

  /**
   * Obtenir les statistiques du service
   */
  getStats() {
    return {
      requests: this.requestCount,
      errors: this.errorCount,
      errorRate: this.requestCount > 0 ? (this.errorCount / this.requestCount * 100).toFixed(2) + '%' : '0%',
      lastErrorTime: this.lastErrorTime,
      cache: cache.getStats(),
      config: {
        apiUrl: CONFIG.API_URL,
        timeout: CONFIG.TIMEOUT,
        maxRetries: CONFIG.MAX_RETRIES
      }
    };
  }

  /**
   * Réinitialiser les statistiques
   */
  resetStats() {
    this.requestCount = 0;
    this.errorCount = 0;
    this.lastErrorTime = null;
    cache.clear();
  }

  /**
   * Tester la connectivité avec différents providers
   */
  async testConnectivity() {
    const results = {
      backend: null,
      openai: null,
      anthropic: null,
      timestamp: new Date().toISOString()
    };

    try {
      // Test du backend
      const healthCheck = await this.checkHealth();
      results.backend = {
        status: healthCheck.healthy ? 'OK' : 'ERREUR',
        details: healthCheck
      };

      // Test OpenAI avec un message court
      try {
        const openaiTest = await this.generateResponse('Test de connectivité', {
          provider: 'openai',
          useCache: false,
          maxTokens: 10
        });
        results.openai = {
          status: 'OK',
          responseLength: openaiTest.content.length,
          model: openaiTest.model
        };
      } catch (error) {
        results.openai = {
          status: 'ERREUR',
          error: error.message,
          type: error.type
        };
      }

      // Test Anthropic avec un message court
      try {
        const anthropicTest = await this.generateResponse('Test de connectivité', {
          provider: 'anthropic',
          useCache: false,
          maxTokens: 10
        });
        results.anthropic = {
          status: 'OK',
          responseLength: anthropicTest.content.length,
          model: anthropicTest.model
        };
      } catch (error) {
        results.anthropic = {
          status: 'ERREUR',
          error: error.message,
          type: error.type
        };
      }

    } catch (error) {
      results.backend = {
        status: 'ERREUR',
        error: error.message
      };
    }

    return results;
  }
}

// Instance singleton du service
const llmService = new LLMService();

// Fonctions d'aide pour compatibilité avec l'ancien code
export const generateResponse = async (message, provider = 'openai', model) => {
  return llmService.generateResponse(message, { provider, model });
};

export const checkProxyServer = async () => {
  const health = await llmService.checkHealth();
  return health.healthy;
};

export const getAvailableModels = () => {
  return {
    OPENAI: ['gpt-4o-mini', 'gpt-3.5-turbo'],
    ANTHROPIC: ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229']
  };
};

export const isProviderConfigured = async (provider) => {
  const health = await llmService.checkHealth();
  return health.services?.[provider] || false;
};

// Exports principaux
export default llmService;
export { LLMError, CONFIG as LLM_CONFIG };