import React, { useContext } from 'react';
import { ChatContext } from '../../context/ChatContext';
import { getAvailableModels, isProviderConfigured } from '../../services/llmService';

const ModelSelector = () => {
  const { 
    selectedProvider, 
    setSelectedProvider, 
    selectedModel, 
    setSelectedModel 
  } = useContext(ChatContext);
  
  const models = getAvailableModels();
  const providers = Object.keys(models).filter(provider => isProviderConfigured(provider));
  
  const handleProviderChange = (e) => {
    const newProvider = e.target.value;
    setSelectedProvider(newProvider);
    // Sélectionner le premier modèle disponible pour ce fournisseur
    setSelectedModel(models[newProvider][0]);
  };
  
  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };
  
  // Si aucun fournisseur n'est configuré, afficher un message
  if (providers.length === 0) {
    return (
      <div className="text-xs text-neutral-500 dark:text-neutral-400 px-2">
        Aucun fournisseur d'IA configuré. Veuillez ajouter vos clés API.
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-2">
      <div className="flex items-center space-x-2">
        <label htmlFor="provider-select" className="text-xs text-neutral-500 dark:text-neutral-400">
          Fournisseur:
        </label>
        <select
          id="provider-select"
          value={selectedProvider}
          onChange={handleProviderChange}
          className="model-selector text-xs py-1"
        >
          {providers.map(provider => (
            <option key={provider} value={provider}>
              {provider === 'OPENAI' ? 'OpenAI' : 
               provider === 'ANTHROPIC' ? 'Anthropic' : 
               provider === 'GEMINI' ? 'Google Gemini' : provider}
            </option>
          ))}
        </select>
      </div>
      
      <div className="flex items-center space-x-2">
        <label htmlFor="model-select" className="text-xs text-neutral-500 dark:text-neutral-400">
          Modèle:
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={handleModelChange}
          className="model-selector text-xs py-1"
        >
          {models[selectedProvider].map(model => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default ModelSelector;
