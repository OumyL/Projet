// server/mcpClient.js - Client MCP pour Windows
const { spawn } = require('child_process');

class MCPClient {
  constructor() {
    this.isConnected = false;
    this.pendingRequests = new Map();
    this.requestId = 0;
    this.mcpProcess = null;
    this.buffer = '';
  }

  async connect() {
    return new Promise((resolve, reject) => {
      console.log('Connexion au MCP-Trader...');
      
      // Démarrer un nouveau processus MCP-Trader
      this.mcpProcess = spawn('uv', ['run', 'mcp-trader'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: 'C:\\Users\\tayab\\Desktop\\Projet PFE\\mcp-trader',
        shell: true
      });

      this.mcpProcess.stdout.on('data', (data) => {
        this.handleData(data.toString());
      });

      this.mcpProcess.stderr.on('data', (data) => {
        const output = data.toString();
        console.log('MCP Info:', output);
        
        // Vérifier si le serveur est prêt
        if (output.includes('Server ready') || output.includes('Starting MCP server')) {
          setTimeout(() => {
            this.initialize()
              .then(() => {
                this.isConnected = true;
                console.log('MCP Client connecté !');
                resolve();
              })
              .catch(reject);
          }, 2000);
        }
      });

      this.mcpProcess.on('error', (error) => {
        console.error('Erreur MCP Process:', error);
        reject(error);
      });

      // Timeout après 15 secondes
      setTimeout(() => {
        if (!this.isConnected) {
          reject(new Error('Timeout: MCP ne répond pas'));
        }
      }, 15000);
    });
  }

  handleData(data) {
    this.buffer += data;
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line);
          this.processMessage(message);
        } catch (e) {
          // Ignorer les lignes non-JSON
        }
      }
    }
  }

  processMessage(message) {
    if (message.id && this.pendingRequests.has(message.id)) {
      const { resolve, reject, timer } = this.pendingRequests.get(message.id);
      this.pendingRequests.delete(message.id);
      clearTimeout(timer);

      if (message.error) {
        reject(new Error(message.error.message || 'Erreur MCP'));
      } else {
        resolve(message.result);
      }
    }
  }

  async initialize() {
    const initRequest = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: {
          name: 'Brain-Agent',
          version: '1.0.0'
        }
      }
    };
    await this.sendRequest(initRequest);

    const initializedNotification = {
    jsonrpc: '2.0',
    method: 'notifications/initialized'
    };
  
     this.mcpProcess.stdin.write(JSON.stringify(initializedNotification) + '\n');
  
  // Attendre un peu avant de considérer l'initialisation complète
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  sendRequest(request) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(request.id);
        reject(new Error(`Timeout pour ${request.method}`));
      }, 15000);

      this.pendingRequests.set(request.id, { resolve, reject, timer });
      
      const requestString = JSON.stringify(request) + '\n';
      this.mcpProcess.stdin.write(requestString);
    });
  }
  
  async listTools() {
  if (!this.isConnected) {
    throw new Error('MCP non connecté');
  }

  const request = {
    jsonrpc: '2.0',
    id: ++this.requestId,
    method: 'tools/list'
  };

  try {
    const result = await this.sendRequest(request);
    return result.tools || result;
  } catch (error) {
    console.error('Erreur listTools:', error);
    throw error;
  }
}
  async callTool(name, arguments_ = {}) {
    if (!this.isConnected) {
      throw new Error('MCP non connecté');
    }

    const request = {
      jsonrpc: '2.0',
      id: ++this.requestId,
      method: 'tools/call',
      params: {
        name,
        arguments: arguments_
      }
    };

    try {
      const result = await this.sendRequest(request);
      // Extraire le contenu texte de la réponse
      if (result.content && Array.isArray(result.content)) {
        return result.content.map(item => item.text || item).join('\n');
      }
      return result;
    } catch (error) {
      console.error(`Erreur outil ${name}:`, error);
      throw error;
    }
  }

  // Méthodes spécifiques pour votre MCP-Trader
  async systemDiagnostic() {
    return this.callTool('system_diagnostic');
  }

  async apiStatusCheck() {
    return this.callTool('api_status_check');
  }
  async analyzeStock(symbol) {
    return this.callTool('analyze_stock', { symbol });
  }
  async relativeStrength(symbol, benchmark = 'SPY') {
    return this.callTool('relative_strength', { symbol, benchmark });
  }
  async volumeProfile(symbol, lookback_days = 60) {
    return this.callTool('volume_profile', { symbol, lookback_days });
  }
  async detectPatterns(symbol) {
    return this.callTool('detect_patterns', { symbol });
  }
  async positionSize(symbol, stopPrice, riskAmount, accountSize) {
    return this.callTool('position_size', {
    symbol,
    stop_price: stopPrice,
    risk_amount: riskAmount,
    account_size: accountSize
    });
  }
  async suggestStops(symbol) {
    return this.callTool('suggest_stops', { symbol });
  }
  async analyzeSentiment(symbol, daysBack = 7) {
    return this.callTool('analyze_sentiment', { 
    symbol, 
    days_back: daysBack 
    });
  }
  async screenMomentumStocks(symbols = 'AAPL,MSFT,GOOGL,AMZN,TSLA') {
    return this.callTool('screen_momentum_stocks', { symbols });
  }
  async marketOverview(symbols = 'AAPL,MSFT,GOOGL,AMZN,TSLA') {
    return this.callTool('market_overview', { symbols });
  }
  async analyzeCrypto(symbol) {
    return this.callTool('analyze_crypto', { symbol });
  }
  async analyzeFundamentals(symbol) {
    return this.callTool('analyze_fundamentals', { symbol });
  }

  async compareCompanies(symbols, metrics = 'pe_ratio,pb_ratio,roe,profit_margin,debt_to_equity') {
    return this.callTool('compare_companies', { symbols, metrics });
  }

  async investmentThesis(symbol) {
    return this.callTool('investment_thesis', { symbol });
  }

  async earningsAnalysis(symbol) {
    return this.callTool('earnings_analysis', { symbol });
  }
  
  async getFinalRecommendation(symbol, analysisType = 'complete') {
    return this.callTool('get_final_recommendation', { 
    symbol, 
    analysis_type: analysisType 
    });
  }

  async getQuickRecommendation(symbol) {
    return this.callTool('quick_recommendation', { symbol });
  }
  disconnect() {
    if (this.mcpProcess) {
      console.log('Déconnexion MCP...');
      this.mcpProcess.kill();
      this.isConnected = false;
    }
  }
}

module.exports = MCPClient;