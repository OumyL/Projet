import React, { useState, useContext, useRef, useEffect } from 'react';
import { ChatContext } from '../../context/ChatContext';
import { getAvailableModels } from '../../services/llmService';

const ModelDropdown = () => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);
  const { selectedProvider, selectedModel, setSelectedProvider, setSelectedModel } = useContext(ChatContext);
  
  const allModels = getAvailableModels();
  const providers = Object.keys(allModels);
  
  // Fermer le dropdown quand on clique ailleurs
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Formater le nom du modèle pour l'affichage
  const formatModelName = (model) => {
    // Extraire juste la partie principale du nom (sans date)
    if (model.includes('-20')) {
      const mainName = model.split('-20')[0];
      return mainName.replace('claude-3-', '').toUpperCase();
    } else if (model.includes('gpt')) {
      return model.replace('gpt-', 'GPT ').toUpperCase();
    }
    return model.toUpperCase();
  };

  // Formater le nom du fournisseur
  const formatProviderName = (provider) => {
    switch(provider) {
      case 'OPENAI': return 'OpenAI';
      case 'ANTHROPIC': return 'Anthropic';
      case 'GEMINI': return 'Google';
      default: return provider;
    }
  };
  
  return (
    <div className="relative z-20" ref={dropdownRef}>
      <button
        className="flex items-center px-3 py-1.5 rounded-full bg-gradient-to-r from-brand-blue/20 to-blue-400/10 dark:from-brand-blue/30 dark:to-blue-500/20 backdrop-blur-sm hover:from-brand-blue/30 hover:to-blue-400/20 dark:hover:from-brand-blue/40 dark:hover:to-blue-500/30 text-xs font-medium text-brand-dark dark:text-brand-white transition-all shadow-sm"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className="mr-1">Modèle:</span>
        <span className="font-bold text-brand-blue">{formatModelName(selectedModel)}</span>
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          className={`h-4 w-4 ml-1 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isOpen && (
        <div className="absolute right-0 z-10 mt-2 w-56 rounded-lg shadow-lg bg-white/90 dark:bg-brand-dark/95 backdrop-blur-md ring-1 ring-black/5 dark:ring-brand-blue/20 focus:outline-none overflow-hidden transition-all duration-150 ease-in-out">
          <div className="max-h-80 overflow-y-auto">
            {providers.map((provider) => (
              <div key={provider} className="py-1">
                <div className="px-3 py-1.5 text-xs font-semibold text-neutral-500 dark:text-brand-gray bg-neutral-100/50 dark:bg-brand-dark/80">
                  {formatProviderName(provider)}
                </div>
                {allModels[provider].map((model) => (
                  <button
                    key={model}
                    className={`block w-full text-left px-4 py-2 text-sm ${
                      provider === selectedProvider && model === selectedModel 
                        ? 'bg-brand-blue/10 dark:bg-brand-blue/20 text-brand-blue dark:text-brand-blue font-medium' 
                        : 'text-gray-700 dark:text-brand-gray hover:bg-gray-100/70 dark:hover:bg-brand-dark/80'
                    }`}
                    onClick={() => {
                      setSelectedProvider(provider);
                      setSelectedModel(model);
                      setIsOpen(false);
                    }}
                  >
                    {formatModelName(model)}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelDropdown;
