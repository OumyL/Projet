import React, { useEffect, useRef, useState, useCallback } from "react";

const TradingViewChart = ({ 
  symbol = "BINANCE:BTCUSDT", 
  interval = "1D", 
  height = 400,
  theme = "dark" 
}) => {
  const containerRef = useRef(null);
  const widgetInstanceRef = useRef(null);
  const scriptRef = useRef(null);
  const mountedRef = useRef(true);
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  // Fonction de nettoyage complète
  const cleanupWidget = useCallback(() => {
    console.log('[TradingView] Cleaning up widget...');
    
    try {
      // Nettoyer l'instance du widget TradingView
      if (widgetInstanceRef.current && typeof widgetInstanceRef.current.destroy === 'function') {
        widgetInstanceRef.current.destroy();
      }
      widgetInstanceRef.current = null;

      // Supprimer le script
      if (scriptRef.current && scriptRef.current.parentNode) {
        scriptRef.current.parentNode.removeChild(scriptRef.current);
      }
      scriptRef.current = null;

      // Nettoyer le conteneur DOM
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    } catch (cleanupError) {
      console.warn('[TradingView] Cleanup error:', cleanupError);
    }
  }, []);

  // Fonction de chargement du widget avec protection contre les re-renders infinis
  const loadWidget = useCallback(async () => {
    // Éviter les chargements multiples
    if (!mountedRef.current || isLoading) {
      return;
    }

    console.log(`[TradingView] Loading widget for ${symbol}...`);
    
    setIsLoading(true);
    setError(null);

    // Nettoyer l'ancien widget avant de créer le nouveau
    cleanupWidget();

    // Délai pour laisser le DOM se stabiliser
    await new Promise(resolve => setTimeout(resolve, 100));

    // Vérifier que le composant est toujours monté
    if (!mountedRef.current || !containerRef.current) {
      return;
    }

    try {
      // Créer un ID unique pour cette instance
      const widgetId = `tradingview_widget_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Créer le conteneur du widget
      const widgetContainer = document.createElement("div");
      widgetContainer.className = "tradingview-widget-container";
      widgetContainer.style.height = typeof height === "string" ? height : `${height}px`;
      
      const widgetDiv = document.createElement("div");
      widgetDiv.id = widgetId;
      widgetDiv.className = "tradingview-widget-container__widget";
      widgetDiv.style.height = "100%";
      
      widgetContainer.appendChild(widgetDiv);
      containerRef.current.appendChild(widgetContainer);

      // Configuration du widget
      const widgetConfig = {
        autosize: true,
        symbol: symbol,
        interval: interval,
        timezone: "Etc/UTC",
        theme: theme,
        style: "1",
        locale: "fr",
        toolbar_bg: "#f1f3f6",
        enable_publishing: false,
        allow_symbol_change: false, // Désactivé pour éviter les conflits
        calendar: false,
        support_host: "https://www.tradingview.com",
        container_id: widgetId,
        // Désactiver les fonctionnalités qui peuvent causer des problèmes
        disabled_features: [
          "use_localstorage_for_settings",
          "volume_force_overlay"
        ],
        enabled_features: [
          "side_toolbar_in_fullscreen_mode"
        ]
      };

      // Créer le script TradingView
      const script = document.createElement("script");
      script.type = "text/javascript";
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.async = true;
      script.innerHTML = JSON.stringify(widgetConfig);
      scriptRef.current = script;

      // Gestionnaires d'événements du script
      script.onload = () => {
        if (!mountedRef.current) return;
        
        console.log(`[TradingView] Script loaded successfully for ${symbol}`);
        
        // Délai pour laisser TradingView initialiser complètement
        setTimeout(() => {
          if (mountedRef.current) {
            setIsLoading(false);
            setRetryCount(0);
          }
        }, 2000);
      };

      script.onerror = (e) => {
        if (!mountedRef.current) return;
        
        console.error('[TradingView] Script loading error:', e);
        setError("Erreur de chargement du graphique TradingView");
        setIsLoading(false);

        // Tentative de retry automatique (max 2 fois)
        if (retryCount < 2) {
          setTimeout(() => {
            if (mountedRef.current) {
              setRetryCount(prev => prev + 1);
              loadWidget();
            }
          }, 3000);
        }
      };

      // Ajouter le script au DOM après un court délai
      setTimeout(() => {
        if (mountedRef.current && widgetContainer.parentNode) {
          widgetContainer.appendChild(script);
        }
      }, 100);

    } catch (err) {
      console.error('[TradingView] Widget creation error:', err);
      if (mountedRef.current) {
        setError("Erreur lors de la création du graphique");
        setIsLoading(false);
      }
    }
  }, [symbol, interval, height, theme, isLoading, retryCount, cleanupWidget]);

  // Effect pour charger le widget quand les props changent
  useEffect(() => {
    mountedRef.current = true;
    
    // Débounce le chargement pour éviter les appels trop fréquents
    const timeoutId = setTimeout(() => {
      if (mountedRef.current) {
        loadWidget();
      }
    }, 500);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [loadWidget]);

  // Effect de nettoyage au démontage
  useEffect(() => {
    return () => {
      console.log('[TradingView] Component unmounting...');
      mountedRef.current = false;
      cleanupWidget();
    };
  }, [cleanupWidget]);

  // Gestionnaire de retry manuel
  const handleRetry = useCallback(() => {
    setRetryCount(0);
    loadWidget();
  }, [loadWidget]);

  // Rendu conditionnel pour les erreurs
  if (error) {
    return (
      <div
        className="flex items-center justify-center bg-gray-800 rounded-lg border border-gray-600"
        style={{ height: typeof height === "string" ? height : `${height}px` }}
      >
        <div className="text-center text-gray-300 p-4">
          <div className="text-red-400 mb-2 text-2xl">⚠️</div>
          <div className="text-sm font-medium mb-2">{error}</div>
          <div className="text-xs text-gray-500 mb-3">
            Symbole: {symbol} | Tentatives: {retryCount}/2
          </div>
          <button 
            onClick={handleRetry}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={retryCount >= 2}
          >
            {retryCount >= 2 ? 'Max tentatives atteintes' : 'Réessayer'}
          </button>
        </div>
      </div>
    );
  }

  // Rendu principal
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-600 relative">
      {/* Indicateur de chargement */}
      {isLoading && (
        <div
          className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10"
          style={{ height: typeof height === "string" ? height : `${height}px` }}
        >
          <div className="text-white text-center">
            <div className="relative mx-auto mb-3">
              <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
            <div className="text-sm text-gray-300 font-medium">
              Chargement du graphique...
            </div>
            <div className="text-xs mt-1 text-gray-500">
              {symbol} • {interval}
            </div>
            {retryCount > 0 && (
              <div className="text-xs mt-1 text-yellow-400">
                Tentative {retryCount + 1}/3
              </div>
            )}
          </div>
        </div>
      )}

      {/* Conteneur du widget */}
      <div
        ref={containerRef}
        style={{
          height: typeof height === "string" ? height : `${height}px`,
          visibility: isLoading ? "hidden" : "visible"
        }}
      />

      {/* Footer TradingView (seulement si chargé) */}
      {!isLoading && !error && (
        <div className="bg-gray-800 px-3 py-1 text-center border-t border-gray-600">
          <span className="text-xs text-gray-400">
            Graphique fourni par{" "}
            <a 
              href="https://tradingview.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              TradingView
            </a>
          </span>
        </div>
      )}
    </div>
  );
};

// Memo pour éviter les re-renders inutiles
export default React.memo(TradingViewChart, (prevProps, nextProps) => {
  return (
    prevProps.symbol === nextProps.symbol &&
    prevProps.interval === nextProps.interval &&
    prevProps.height === nextProps.height &&
    prevProps.theme === nextProps.theme
  );
});