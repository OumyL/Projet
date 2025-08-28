import React, { useEffect, useRef, useState, useCallback } from "react";

const TradingViewChart = ({ symbol = "BINANCE:BTCUSDT", interval = "1D", height = 400 }) => {
  const containerRef = useRef(null);
  const widgetRef = useRef(null);
  const scriptRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (scriptRef.current) {
      try {
        if (scriptRef.current.parentNode) {
          scriptRef.current.parentNode.removeChild(scriptRef.current);
        }
      } catch (e) {
        console.warn('Script cleanup failed:', e);
      }
      scriptRef.current = null;
    }

    if (containerRef.current) {
      containerRef.current.innerHTML = "";
    }

    if (widgetRef.current) {
      widgetRef.current = null;
    }
  }, []);

  const loadWidget = useCallback(async () => {
    if (!containerRef.current) return;

    setIsLoading(true);
    setError(null);
    cleanup();

    try {
      await new Promise(resolve => setTimeout(resolve, 100));

      const widgetContainer = document.createElement("div");
      widgetContainer.className = "tradingview-widget-container";
      
      const widgetDiv = document.createElement("div");
      const widgetId = `tradingview_${symbol.replace(":", "_")}_${Date.now()}`;
      widgetDiv.id = widgetId;
      widgetDiv.className = "tradingview-widget-container__widget";
      widgetDiv.style.height = typeof height === "string" ? height : `${height}px`;
      
      widgetContainer.appendChild(widgetDiv);
      containerRef.current.appendChild(widgetContainer);
      
      // Store reference for cleanup
      widgetRef.current = widgetDiv;

      const script = document.createElement("script");
      script.type = "text/javascript";
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.async = true;
      scriptRef.current = script;

      const config = {
        autosize: true,
        symbol: symbol,
        interval: interval,
        timezone: "Etc/UTC",
        theme: "dark",
        style: "1",
        locale: "fr",
        enable_publishing: false,
        allow_symbol_change: true,
        calendar: false,
        support_host: "https://www.tradingview.com",
        container_id: widgetId,
      };

      script.innerHTML = JSON.stringify(config);

      script.onload = () => {
        console.log(`TradingView widget loaded for ${symbol}`);
        setIsLoading(false);
      };

      script.onerror = (e) => {
        console.error("TradingView script failed to load:", e);
        setError("Erreur de chargement TradingView");
        setIsLoading(false);
      };

      // Append script after short delay
      setTimeout(() => {
        if (widgetContainer.parentNode) {
          widgetContainer.appendChild(script);
        }
      }, 50);

    } catch (err) {
      console.error("Error creating TradingView widget:", err);
      setError("Erreur lors de la création du graphique");
      setIsLoading(false);
    }
  }, [symbol, interval, height, cleanup]);

  useEffect(() => {
    loadWidget();
    return cleanup;
  }, [loadWidget, cleanup]);

  if (error) {
    return (
      <div
        className="flex items-center justify-center bg-gray-800 rounded-lg border border-gray-600"
        style={{ height: typeof height === "string" ? height : `${height}px` }}
      >
        <div className="text-center text-gray-300">
          <div className="text-red-400 mb-2">⚠️</div>
          <div className="text-sm">{error}</div>
          <div className="text-xs mt-2 text-gray-500">Symbole: {symbol}</div>
          <button 
            onClick={loadWidget}
            className="mt-2 px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
          >
            Réessayer
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-600">
      {isLoading && (
        <div
          className="flex items-center justify-center bg-gray-800"
          style={{ height: typeof height === "string" ? height : `${height}px` }}
        >
          <div className="text-white text-center">
            <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
            <div className="text-sm text-gray-300">Chargement du graphique...</div>
            <div className="text-xs mt-1 text-gray-500">{symbol}</div>
          </div>
        </div>
      )}

      <div
        ref={containerRef}
        style={{
          height: typeof height === "string" ? height : `${height}px`,
          display: isLoading ? "none" : "block",
        }}
      />

      {!isLoading && !error && (
        <div className="tradingview-widget-copyright text-center py-1 bg-gray-800">
          <span className="text-xs text-gray-400">Graphique par TradingView</span>
        </div>
      )}
    </div>
  );
};

export default TradingViewChart;