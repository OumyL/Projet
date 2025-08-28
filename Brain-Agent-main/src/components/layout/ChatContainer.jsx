import React, { useContext, useState, useEffect, useCallback } from 'react';
import { ChatContext } from '../../context/ChatContext';
import { ThemeContext } from '../../context/ThemeContext';
import ChatMessages from '../chat/ChatMessages';
import MessageInput from '../chat/MessageInput';
import ModelDropdown from '../ui/ModelDropdown';
import TradingViewChart from '../chart/TradingViewChart';

const ChatContainer = ({ onOpenSidebar, onToggleSidebar, isSidebarVisible }) => {
  const { activeConversation } = useContext(ChatContext);
  const { isDark } = useContext(ThemeContext);
  
  // État pour gérer l'affichage du graphique
  const [showChart, setShowChart] = useState(false);
  const [currentSymbol, setCurrentSymbol] = useState(null);
  const [chartInterval, setChartInterval] = useState("1D");

  // Mapping des symboles étendu
  const SYMBOL_MAPPING = {
    // === CRYPTOMONNAIES ===
    'BTC': 'BINANCE:BTCUSDT',
    'ETH': 'BINANCE:ETHUSDT',
    'BNB': 'BINANCE:BNBUSDT',
    'ADA': 'BINANCE:ADAUSDT',
    'SOL': 'BINANCE:SOLUSDT',
    'XRP': 'BINANCE:XRPUSDT',
    'DOT': 'BINANCE:DOTUSDT',
    'AVAX': 'BINANCE:AVAXUSDT',
    'LINK': 'BINANCE:LINKUSDT',
    'MATIC': 'BINANCE:MATICUSDT',
    'UNI': 'BINANCE:UNIUSDT',
    'LTC': 'BINANCE:LTCUSDT',
    'DOGE': 'BINANCE:DOGEUSDT',
    'SHIB': 'BINANCE:SHIBUSDT',

    // === ACTIONS TECH ===
    'AAPL': 'NASDAQ:AAPL',
    'MSFT': 'NASDAQ:MSFT',
    'GOOGL': 'NASDAQ:GOOGL',
    'AMZN': 'NASDAQ:AMZN',
    'TSLA': 'NASDAQ:TSLA',
    'META': 'NASDAQ:META',
    'NFLX': 'NASDAQ:NFLX',
    'NVDA': 'NASDAQ:NVDA',
    'AMD': 'NASDAQ:AMD',
    'INTC': 'NASDAQ:INTC',

    // === ACTIONS TRADITIONNELLES ===
    'JPM': 'NYSE:JPM',
    'BAC': 'NYSE:BAC',
    'V': 'NYSE:V',
    'MA': 'NYSE:MA',
    'JNJ': 'NYSE:JNJ',
    'PG': 'NYSE:PG',
    'KO': 'NYSE:KO',
    'DIS': 'NYSE:DIS',
    'NKE': 'NYSE:NKE',
    'MCD': 'NYSE:MCD',
    'WMT': 'NYSE:WMT',

    // === INDICES ===
    'SPY': 'AMEX:SPY',
    'QQQ': 'NASDAQ:QQQ',
    'SPX': 'TVC:SPX',
    'NDX': 'TVC:NDX',
    'DJI': 'TVC:DJI',

    // === MATIÈRES PREMIÈRES ===
    'GOLD': 'TVC:GOLD',
    'SILVER': 'TVC:SILVER',
    'OIL': 'TVC:USOIL',

    // === DEVISES ===
    'EURUSD': 'FX:EURUSD',
    'GBPUSD': 'FX:GBPUSD',
    'USDJPY': 'FX:USDJPY'
  };

  // Fonction améliorée pour détecter les symboles dans le DERNIER message uniquement
  const detectSymbolInLastMessage = useCallback(() => {
    if (!activeConversation || !activeConversation.messages.length) return null;
    
    // Prendre SEULEMENT le dernier message utilisateur
    const messages = activeConversation.messages;
    const lastUserMessage = [...messages].reverse().find(msg => msg.role === 'user');
    
    if (!lastUserMessage) return null;

    console.log('🔍 Analysing last user message:', lastUserMessage.content);
    
    const upperContent = lastUserMessage.content.toUpperCase();
    
    // Chercher les symboles dans le dernier message
    for (const [ticker, tvSymbol] of Object.entries(SYMBOL_MAPPING)) {
      if (upperContent.includes(ticker)) {
        const detected = { ticker, symbol: tvSymbol };
        console.log('✅ Symbol detected:', detected);
        return detected;
      }
    }
    
    console.log('❌ No symbol detected in last message');
    return null;
  }, [activeConversation?.messages, SYMBOL_MAPPING]);

  // Effet pour détecter automatiquement les symboles - DÉCLENCHÉ SEULEMENT SUR NOUVEAU MESSAGE
  useEffect(() => {
    if (!activeConversation?.messages?.length) return;
    
    const detectedSymbol = detectSymbolInLastMessage();
    if (detectedSymbol) {
      console.log('🎯 Updating chart to:', detectedSymbol);
      setCurrentSymbol(detectedSymbol);
      setShowChart(true);
    }
  }, [activeConversation?.messages?.length, detectSymbolInLastMessage]); // Dépendance sur la LONGUEUR pour éviter les updates inutiles

  const intervals = [
    { value: "1", label: "1m" },
    { value: "5", label: "5m" },
    { value: "15", label: "15m" },
    { value: "1H", label: "1H" },
    { value: "4H", label: "4H" },
    { value: "1D", label: "1D" },
    { value: "1W", label: "1W" }
  ];

  return (
    <div className="flex h-full flex-1 bg-white dark:bg-brand-dark overflow-hidden">
      {/* Zone de chat principale */}
      <div className={`flex flex-col ${showChart ? 'w-1/2' : 'w-full'} transition-all duration-300`}>
        {/* En-tête du chat */}
        <div className="p-4 border-b border-neutral-200/70 dark:border-brand-blue/10 bg-white dark:bg-brand-dark z-10 flex items-center justify-between">
          <div className="flex items-center">
            {/* Bouton pour basculer l'affichage de la sidebar */}
            <button 
              onClick={onToggleSidebar}
              className="mr-3 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-brand-dark/80 text-neutral-600 dark:text-brand-gray transition-colors hidden md:block"
              aria-label={isSidebarVisible ? "Masquer les conversations" : "Afficher les conversations"}
            >
              {isSidebarVisible ? (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                </svg>
              )}
            </button>

            {/* Bouton pour ouvrir la sidebar (visible uniquement en mobile) */}
            <button 
              onClick={onOpenSidebar}
              className="mr-3 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-brand-dark/80 text-neutral-600 dark:text-brand-gray transition-colors md:hidden"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
              </svg>
            </button>
            
            {/* Titre de la conversation */}
            <h2 className="text-lg font-medium text-brand-dark dark:text-brand-white">
              {activeConversation ? activeConversation.title : 'BRAIN'}
            </h2>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Bouton pour afficher/masquer le graphique */}
            <button
              onClick={() => setShowChart(!showChart)}
              className={`p-2 rounded-lg transition-colors ${
                showChart 
                  ? 'bg-brand-blue/20 text-brand-blue' 
                  : 'text-neutral-600 dark:text-brand-gray hover:bg-neutral-100 dark:hover:bg-brand-dark/80'
              }`}
              aria-label={showChart ? "Masquer le graphique" : "Afficher le graphique"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
              </svg>
            </button>
            
            {/* Sélecteur de modèle */}
            <ModelDropdown />
          </div>
        </div>
        
        {/* Zone des messages */}
        <ChatMessages />
        
        {/* Zone de saisie de message */}
        <MessageInput />
      </div>

      {/* Zone graphique (conditionnelle) */}
      {showChart && (
        <div className="w-1/2 border-l border-neutral-200/70 dark:border-brand-blue/10 bg-white dark:bg-brand-dark flex flex-col">
          {/* En-tête du graphique */}
          <div className="p-4 border-b border-neutral-200/70 dark:border-brand-blue/10">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-brand-dark dark:text-brand-white">
                Graphique {currentSymbol?.ticker || 'BTC'}
              </h3>
              
              <div className="flex items-center space-x-2">
                {/* Sélecteur d'intervalle */}
                <select
                  value={chartInterval}
                  onChange={(e) => setChartInterval(e.target.value)}
                  className="px-2 py-1 text-xs rounded bg-neutral-100 dark:bg-brand-dark border border-neutral-300 dark:border-brand-blue/30 text-neutral-700 dark:text-brand-gray"
                >
                  {intervals.map(interval => (
                    <option key={interval.value} value={interval.value}>
                      {interval.label}
                    </option>
                  ))}
                </select>
                
                {/* Bouton fermer */}
                <button
                  onClick={() => setShowChart(false)}
                  className="p-1 rounded text-neutral-600 dark:text-brand-gray hover:bg-neutral-100 dark:hover:bg-brand-dark/80"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
          
          {/* Graphique TradingView avec key pour forcer le re-render */}
          <div className="flex-1 p-4">
            <TradingViewChart 
              key={`${currentSymbol?.symbol}-${chartInterval}`} // Force le re-render quand symbole change
              symbol={currentSymbol?.symbol || 'BINANCE:BTCUSDT'}
              interval={chartInterval}
              height="calc(100vh - 200px)"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatContainer;