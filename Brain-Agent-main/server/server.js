const MCPClient = require('./mcpClient');
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = 3001;

let mcpClient = null;
let mcpConnected = false;
let mcpInitializing = false;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// AMÉLIORATION : Fonction d'initialisation MCP avec retry et gestion d'erreurs
async function initializeMCP() {
  if (mcpInitializing) {
    console.log('MCP initialization already in progress...');
    return;
  }

  mcpInitializing = true;
  const maxRetries = 3;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Initialisation du client MCP (tentative ${attempt}/${maxRetries})...`);
      mcpClient = new MCPClient();
      await mcpClient.connect();
      mcpConnected = true;
      mcpInitializing = false;
      console.log('✅ MCP Client initialisé avec succès');
      return;
    } catch (error) {
      console.error(`❌ Erreur initialisation MCP (tentative ${attempt}):`, error.message);
      
      if (mcpClient) {
        mcpClient.disconnect();
        mcpClient = null;
      }
      
      if (attempt === maxRetries) {
        mcpConnected = false;
        mcpInitializing = false;
        console.error('🔴 MCP initialization failed after all attempts');
        // Ne pas faire planter le serveur, continuer sans MCP
      } else {
        // Attendre avant retry
        console.log(`⏳ Attente de ${attempt * 5}s avant retry...`);
        await new Promise(resolve => setTimeout(resolve, attempt * 5000));
      }
    }
  }
}

// AMÉLIORATION : Route de proxy pour l'API Anthropic avec meilleure gestion d'erreurs
app.post('/api/anthropic', async (req, res) => {
  try {
    const apiKey = req.headers['x-api-key'];
    if (!apiKey) {
      return res.status(400).json({ error: { message: 'API key is required' } });
    }

    // Timeout plus court pour éviter les blocages
    const response = await axios.post('https://api.anthropic.com/v1/messages', req.body, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      timeout: 30000 // 30 secondes
    });

    res.json(response.data);
    
  } catch (error) {
    const statusCode = error.response?.status || 500;
    const errorData = error.response?.data || {};
    
    // Log détaillé pour debugging
    console.error('Anthropic API error:', {
      status: statusCode,
      message: errorData.error?.message || error.message,
      type: errorData.error?.type || 'unknown'
    });
    
    // Réponse structurée pour le client
    res.status(statusCode).json({
      error: {
        message: errorData.error?.message || error.message,
        type: errorData.error?.type || 'api_error',
        status: statusCode,
        details: statusCode === 400 && errorData.error?.message?.includes('credit') ? 
          'Insufficient credits - please add credits to your Anthropic account' : null
      }
    });
  }
});

// AMÉLIORATION : Route de proxy pour l'API OpenAI avec retry
app.post('/api/openai', async (req, res) => {
  const maxRetries = 2;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const apiKey = req.headers['authorization']?.replace('Bearer ', '');
      if (!apiKey) {
        return res.status(400).json({ error: { message: 'API key is required' } });
      }

      const response = await axios.post('https://api.openai.com/v1/chat/completions', req.body, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        timeout: 30000 // 30 secondes
      });

      res.json(response.data);
      return; // Sortir du loop de retry en cas de succès
      
    } catch (error) {
      const statusCode = error.response?.status || 500;
      const errorData = error.response?.data || {};
      
      console.error(`OpenAI API error (attempt ${attempt}):`, {
        status: statusCode,
        message: errorData.error?.message || error.message
      });
      
      // Retry pour certaines erreurs temporaires
      if (attempt < maxRetries && (statusCode === 429 || statusCode >= 500)) {
        console.log(`⏳ Retrying OpenAI call in ${attempt * 2}s...`);
        await new Promise(resolve => setTimeout(resolve, attempt * 2000));
        continue;
      }
      
      // Erreur finale
      res.status(statusCode).json({
        error: {
          message: errorData.error?.message || error.message,
          type: errorData.error?.type || 'api_error',
          status: statusCode
        }
      });
      return;
    }
  }
});

// Route de test améliorée
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'API proxy server is running',
    mcp_connected: mcpConnected,
    timestamp: new Date().toISOString()
  });
});

// AMÉLIORATION : Routes MCP avec gestion d'erreurs robuste
app.get('/api/mcp/status', (req, res) => {
  res.json({
    connected: mcpConnected,
    initializing: mcpInitializing,
    client_available: !!mcpClient,
    timestamp: new Date().toISOString()
  });
});

app.post('/api/mcp/analyze', async (req, res) => {
  // Vérifications préliminaires
  if (mcpInitializing) {
    return res.status(503).json({ 
      success: false, 
      error: 'MCP en cours d\'initialisation, veuillez patienter...' 
    });
  }

  if (!mcpConnected || !mcpClient) {
    // Tentative de reconnexion automatique
    console.log('🔄 Tentative de reconnexion MCP automatique...');
    try {
      await initializeMCP();
      if (!mcpConnected) {
        return res.status(503).json({ 
          success: false, 
          error: 'MCP non connecté et reconnexion échouée' 
        });
      }
    } catch (error) {
      return res.status(503).json({ 
        success: false, 
        error: 'Reconnexion MCP échouée' 
      });
    }
  }

  const { symbol } = req.body;
  if (!symbol) {
    return res.status(400).json({ 
      success: false, 
      error: 'Symbol requis' 
    });
  }

  try {
    console.log(`📊 Analyse MCP pour ${symbol}...`);
    
    // Timeout pour éviter les blocages
    const analysisPromise = mcpClient.analyzeStock(symbol.toUpperCase());
    const timeoutPromise = new Promise((_, reject) => 
      setTimeout(() => reject(new Error('MCP analysis timeout')), 30000)
    );
    
    const result = await Promise.race([analysisPromise, timeoutPromise]);
    
    res.json({
      success: true,
      data: result,
      symbol: symbol.toUpperCase(),
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error(`❌ Erreur analyse MCP ${symbol}:`, error.message);
    
    // Gérer différents types d'erreurs
    let errorMessage = error.message;
    let shouldReconnect = false;
    
    if (error.message.includes('timeout')) {
      errorMessage = 'Analyse trop longue (>30s) - réessayez avec moins de données';
    } else if (error.message.includes('connection') || error.message.includes('disconnect')) {
      errorMessage = 'Connexion MCP perdue - reconnexion automatique...';
      shouldReconnect = true;
    } else if (error.message.includes('rate') || error.message.includes('limit')) {
      errorMessage = 'Limite API atteinte - réessayez dans 1-2 minutes';
    }
    
    res.status(500).json({
      success: false,
      error: errorMessage,
      error_type: error.constructor.name,
      timestamp: new Date().toISOString()
    });
    
    // Reconnexion automatique si nécessaire
    if (shouldReconnect) {
      console.log('🔄 Déclenchement reconnexion MCP automatique...');
      mcpConnected = false;
      setImmediate(() => initializeMCP()); // Asynchrone pour ne pas bloquer la réponse
    }
  }
});

// NOUVEAU : Route pour lister les outils MCP disponibles
app.get('/api/mcp/tools', async (req, res) => {
  if (!mcpConnected || !mcpClient) {
    return res.status(503).json({ 
      success: false, 
      error: 'MCP non connecté' 
    });
  }

  try {
    const tools = await mcpClient.listTools();
    res.json({
      success: true,
      tools: tools,
      count: tools.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Erreur liste outils MCP:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// NOUVEAU : Route générique pour appeler n'importe quel outil MCP
app.post('/api/mcp/tool/:toolName', async (req, res) => {
  if (!mcpConnected || !mcpClient) {
    return res.status(503).json({ 
      success: false, 
      error: 'MCP non connecté' 
    });
  }

  const { toolName } = req.params;
  const args = req.body || {};

  try {
    console.log(`🔧 Appel outil MCP: ${toolName}`, args);
    
    // Vérifier que l'outil existe dans mcpClient
    if (typeof mcpClient[toolName] !== 'function') {
      return res.status(404).json({
        success: false,
        error: `Outil MCP '${toolName}' non trouvé`
      });
    }
    
    const result = await mcpClient[toolName](...Object.values(args));
    
    res.json({
      success: true,
      data: result,
      tool: toolName,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error(`❌ Erreur outil MCP ${toolName}:`, error);
    res.status(500).json({
      success: false,
      error: error.message,
      tool: toolName
    });
  }
});

// AMÉLIORATION : Gestion graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 SIGTERM reçu, arrêt graceful...');
  if (mcpClient) {
    mcpClient.disconnect();
  }
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('🛑 SIGINT reçu, arrêt graceful...');
  if (mcpClient) {
    mcpClient.disconnect();
  }
  process.exit(0);
});

// Initialiser MCP au démarrage avec retry automatique
(async () => {
  console.log('🚀 Démarrage serveur proxy...');
  await initializeMCP();
  
  // Si MCP échoue, programmer des tentatives de reconnexion
  if (!mcpConnected) {
    console.log('⏰ Programmation reconnexions MCP automatiques...');
    setInterval(async () => {
      if (!mcpConnected && !mcpInitializing) {
        console.log('🔄 Tentative reconnexion MCP programmée...');
        await initializeMCP();
      }
    }, 60000); // Tentative toutes les minutes
  }
})();

// Démarrer le serveur
const server = app.listen(port, () => {
  console.log(`✅ API proxy server running on http://localhost:${port}`);
  console.log(`📊 MCP Status: ${mcpConnected ? 'Connected' : 'Disconnected'}`);
});