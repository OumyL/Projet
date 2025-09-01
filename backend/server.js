// backend/server.js - Serveur sécurisé pour BRAIN Trading Assistant
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const axios = require('axios');
const path = require('path');

const app = express();
const port = process.env.PORT || 3001;

// ============== MIDDLEWARE DE SÉCURITÉ ==============

// Helmet pour la sécurité des headers HTTP
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "https://s3.tradingview.com", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https://api.openai.com", "https://api.anthropic.com"],
      frameSrc: ["'self'", "https://s.tradingview.com"],
    },
  },
}));

// CORS sécurisé
app.use(cors({
  origin: function (origin, callback) {
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [
      'http://localhost:3000',
      'http://127.0.0.1:3000'
    ];
    
    // Permettre les requêtes sans origin (mobile, Postman, etc.)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Non autorisé par CORS'));
    }
  },
  credentials: true
}));

// Rate limiting par IP
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // max 100 requêtes par IP par fenêtre de 15min
  message: {
    error: {
      message: 'Trop de requêtes depuis cette IP. Réessayez dans 15 minutes.',
      type: 'rate_limit_exceeded'
    }
  },
  standardHeaders: true,
  legacyHeaders: false,
});

// Appliquer le rate limiting seulement aux endpoints API
app.use('/api/', limiter);

// Parser JSON avec limite de taille
app.use(express.json({ limit: '10mb' }));

// ============== VALIDATION DES ENTRÉES ==============

const validateChatRequest = [
  body('messages')
    .isArray({ min: 1, max: 20 })
    .withMessage('Messages doit être un array de 1 à 20 éléments'),
  
  body('messages.*.role')
    .isIn(['user', 'assistant', 'system'])
    .withMessage('Role invalide'),
    
  body('messages.*.content')
    .isString()
    .isLength({ min: 1, max: 10000 })
    .withMessage('Contenu requis (1-10000 caractères)'),
    
  body('model')
    .optional()
    .isString()
    .isIn(['gpt-4o-mini', 'gpt-3.5-turbo', 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229'])
    .withMessage('Modèle invalide'),
    
  body('temperature')
    .optional()
    .isFloat({ min: 0, max: 2 })
    .withMessage('Température doit être entre 0 et 2'),
    
  body('max_tokens')
    .optional()
    .isInt({ min: 1, max: 4000 })
    .withMessage('Max tokens doit être entre 1 et 4000')
];

// ============== ENDPOINTS SÉCURISÉS ==============

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'opérationnel',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    services: {
      openai: !!process.env.OPENAI_API_KEY,
      anthropic: !!process.env.ANTHROPIC_API_KEY,
      mcp_server: process.env.MCP_SERVER_URL || 'localhost:8000'
    },
    uptime: process.uptime()
  });
});

// Endpoint OpenAI sécurisé
app.post('/api/openai', validateChatRequest, async (req, res) => {
  try {
    // Vérification des erreurs de validation
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: {
          message: 'Données de requête invalides',
          type: 'validation_error',
          details: errors.array()
        }
      });
    }

    // Vérification de la clé API
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'Service temporairement indisponible',
          type: 'service_unavailable'
        }
      });
    }

    // Préparer la requête pour OpenAI
    const requestData = {
      model: req.body.model || 'gpt-4o-mini',
      messages: req.body.messages,
      temperature: req.body.temperature || 0.7,
      max_tokens: req.body.max_tokens || 2000,
      stream: false
    };

    console.log(`[OpenAI] Requête: ${requestData.model}, ${requestData.messages.length} messages`);

    // Appel à l'API OpenAI
    const response = await axios.post('https://api.openai.com/v1/chat/completions', requestData, {
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
        'User-Agent': 'BRAIN-Trading-Assistant/1.0'
      },
      timeout: 30000
    });

    // Log de succès
    console.log(`[OpenAI] Succès: ${response.data.usage?.total_tokens || 'N/A'} tokens utilisés`);

    res.json(response.data);

  } catch (error) {
    console.error('[OpenAI] Erreur:', error.response?.data || error.message);
    
    // Gestion fine des erreurs
    if (error.response) {
      const status = error.response.status;
      const errorData = error.response.data;
      
      let errorMessage = 'Erreur du service OpenAI';
      let errorType = 'api_error';
      
      switch (status) {
        case 401:
          errorMessage = 'Configuration du service incorrecte';
          errorType = 'authentication_error';
          break;
        case 429:
          errorMessage = 'Limite de taux atteinte. Réessayez dans quelques minutes.';
          errorType = 'rate_limit_exceeded';
          break;
        case 400:
          errorMessage = errorData.error?.message || 'Requête invalide';
          errorType = 'invalid_request';
          break;
        case 500:
        case 502:
        case 503:
          errorMessage = 'Service OpenAI temporairement indisponible';
          errorType = 'service_unavailable';
          break;
      }
      
      return res.status(status >= 500 ? 502 : status).json({
        error: {
          message: errorMessage,
          type: errorType,
          provider: 'openai'
        }
      });
    }
    
    // Erreur réseau ou timeout
    res.status(504).json({
      error: {
        message: 'Timeout ou erreur réseau avec OpenAI',
        type: 'network_error',
        provider: 'openai'
      }
    });
  }
});

// Endpoint Anthropic sécurisé
app.post('/api/anthropic', validateChatRequest, async (req, res) => {
  try {
    // Vérification des erreurs de validation
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: {
          message: 'Données de requête invalides',
          type: 'validation_error',
          details: errors.array()
        }
      });
    }

    // Vérification de la clé API
    if (!process.env.ANTHROPIC_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'Service temporairement indisponible',
          type: 'service_unavailable'
        }
      });
    }

    // Préparer la requête pour Anthropic
    const requestData = {
      model: req.body.model || 'claude-3-haiku-20240307',
      messages: req.body.messages,
      max_tokens: req.body.max_tokens || 1000,
      temperature: req.body.temperature || 0.7,
      system: "Vous êtes BRAIN, un assistant IA français spécialisé en finance et trading. Répondez de manière professionnelle et précise."
    };

    console.log(`[Anthropic] Requête: ${requestData.model}, ${requestData.messages.length} messages`);

    // Appel à l'API Anthropic
    const response = await axios.post('https://api.anthropic.com/v1/messages', requestData, {
      headers: {
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01',
        'User-Agent': 'BRAIN-Trading-Assistant/1.0'
      },
      timeout: 30000
    });

    console.log(`[Anthropic] Succès: ${response.data.usage?.output_tokens || 'N/A'} tokens générés`);

    res.json(response.data);

  } catch (error) {
    console.error('[Anthropic] Erreur:', error.response?.data || error.message);
    
    if (error.response) {
      const status = error.response.status;
      const errorData = error.response.data;
      
      let errorMessage = 'Erreur du service Anthropic';
      let errorType = 'api_error';
      
      switch (status) {
        case 401:
          errorMessage = 'Configuration du service incorrecte';
          errorType = 'authentication_error';
          break;
        case 429:
          errorMessage = 'Limite de taux atteinte. Réessayez dans quelques minutes.';
          errorType = 'rate_limit_exceeded';
          break;
        case 400:
          errorMessage = errorData.error?.message || 'Requête invalide';
          errorType = 'invalid_request';
          break;
      }
      
      return res.status(status >= 500 ? 502 : status).json({
        error: {
          message: errorMessage,
          type: errorType,
          provider: 'anthropic'
        }
      });
    }
    
    res.status(504).json({
      error: {
        message: 'Timeout ou erreur réseau avec Anthropic',
        type: 'network_error',
        provider: 'anthropic'
      }
    });
  }
});

// Endpoint pour le serveur MCP
app.post('/api/mcp/:action', async (req, res) => {
  const { action } = req.params;
  const mcpServerUrl = process.env.MCP_SERVER_URL || 'http://localhost:8000';
  
  try {
    console.log(`[MCP] Action: ${action}`);
    
    const response = await axios.post(`${mcpServerUrl}/api/mcp/${action}`, req.body, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 15000
    });
    
    res.json(response.data);
    
  } catch (error) {
    console.error(`[MCP] Erreur ${action}:`, error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: `Erreur MCP ${action}`,
        type: 'mcp_error',
        details: error.response?.data || error.message
      }
    });
  }
});

// ============== GESTION D'ERREURS GLOBALE ==============

// Middleware de gestion d'erreurs
app.use((err, req, res, next) => {
  console.error('Erreur serveur non gérée:', err);
  
  res.status(500).json({
    error: {
      message: 'Erreur interne du serveur',
      type: 'internal_server_error',
      timestamp: new Date().toISOString()
    }
  });
});

// Handler pour les routes non trouvées
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Route ${req.method} ${req.path} non trouvée`,
      type: 'route_not_found'
    }
  });
});

// ============== DÉMARRAGE DU SERVEUR ==============

// Fonction de validation de la configuration au démarrage
function validateConfiguration() {
  const requiredEnvVars = [];
  
  if (!process.env.OPENAI_API_KEY) {
    requiredEnvVars.push('OPENAI_API_KEY');
  }
  
  if (!process.env.ANTHROPIC_API_KEY) {
    requiredEnvVars.push('ANTHROPIC_API_KEY');
  }
  
  if (requiredEnvVars.length > 0) {
    console.warn('⚠️  Variables d\'environnement manquantes:', requiredEnvVars.join(', '));
    console.warn('⚠️  Certains services ne seront pas disponibles');
  }
  
  return requiredEnvVars.length === 0;
}

// Démarrage du serveur
app.listen(port, () => {
  console.log('\n🚀 =================================');
  console.log(`🚀 BRAIN Trading Backend démarré`);
  console.log(`🚀 Port: ${port}`);
  console.log(`🚀 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log('🚀 =================================');
  
  const configValid = validateConfiguration();
  
  if (configValid) {
    console.log('✅ Configuration complète - Tous les services disponibles');
  } else {
    console.log('⚠️  Configuration partielle - Vérifiez le fichier .env');
  }
  
  console.log(`\n📊 Health check: http://localhost:${port}/api/health`);
  console.log(`🔒 Endpoints sécurisés:`);
  console.log(`   • POST /api/openai - Service OpenAI`);
  console.log(`   • POST /api/anthropic - Service Anthropic`);
  console.log(`   • POST /api/mcp/:action - Serveur MCP\n`);
  
  // Test de connectivité au démarrage
  if (process.env.NODE_ENV !== 'production') {
    console.log('🔧 Mode développement - Tests de connectivité désactivés');
  }
});

// Gestion des signaux de fermeture
process.on('SIGTERM', () => {
  console.log('\n🛑 Signal SIGTERM reçu - Arrêt gracieux du serveur...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('\n🛑 Signal SIGINT reçu - Arrêt gracieux du serveur...');
  process.exit(0);
});

module.exports = app;