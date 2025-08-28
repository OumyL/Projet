import React, { createContext, useState, useEffect } from 'react';
import { mockConversations } from '../utils/mockData';
import { generateResponse } from '../services/llmService';

export const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  
  // États pour le choix du modèle
  const [selectedProvider, setSelectedProvider] = useState('ANTHROPIC');
  const [selectedModel, setSelectedModel] = useState('claude-3-haiku-20240307');

  // Charger les conversations depuis localStorage ou utiliser les données fictives
  useEffect(() => {
    const savedConversations = localStorage.getItem('conversations');
    
    if (savedConversations) {
      const parsedConversations = JSON.parse(savedConversations);
      setConversations(parsedConversations);
      
      // Si une conversation active était sauvegardée, la restaurer
      const activeId = localStorage.getItem('activeConversationId');
      if (activeId) {
        const active = parsedConversations.find(c => c.id.toString() === activeId);
        if (active) {
          setActiveConversation(active);
        } else if (parsedConversations.length > 0) {
          setActiveConversation(parsedConversations[0]);
        }
      } else if (parsedConversations.length > 0) {
        setActiveConversation(parsedConversations[0]);
      }
    } else {
      // Mettre à jour les conversations mock pour refléter BRAIN au lieu de Claude
      const updatedMockConversations = mockConversations.map(conv => ({
        ...conv,
        title: conv.title.replace('Claude', 'BRAIN'),
        messages: conv.messages.map(msg => ({
          ...msg,
          content: msg.content.replace(/Claude/g, 'BRAIN')
        }))
      }));
      
      setConversations(updatedMockConversations);
      if (updatedMockConversations.length > 0) {
        setActiveConversation(updatedMockConversations[0]);
      }
    }
    
    // Charger les préférences de modèle
    const savedProvider = localStorage.getItem('selectedProvider');
    const savedModel = localStorage.getItem('selectedModel');
    
    if (savedProvider) setSelectedProvider(savedProvider);
    if (savedModel) setSelectedModel(savedModel);
  }, []);
  
  // Sauvegarder les conversations et la conversation active dans localStorage
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('conversations', JSON.stringify(conversations));
    }
    
    if (activeConversation) {
      localStorage.setItem('activeConversationId', activeConversation.id);
    }
  }, [conversations, activeConversation]);
  
  // Sauvegarder les préférences de modèle
  useEffect(() => {
    localStorage.setItem('selectedProvider', selectedProvider);
    localStorage.setItem('selectedModel', selectedModel);
  }, [selectedProvider, selectedModel]);

  // Envoyer un message avec appel à l'API du modèle sélectionné
  const sendMessage = async (content) => {
    if (!content.trim()) return;
    
    // Créer un nouveau message utilisateur
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    };
    
    // Ajouter le message à la conversation active
    const updatedConversation = {
      ...activeConversation,
      messages: [...activeConversation.messages, userMessage],
      lastMessage: content,
      date: new Date().toISOString()
    };
    
    // Mettre à jour la conversation active
    setActiveConversation(updatedConversation);
    
    // Mettre à jour la liste des conversations
    const updatedConversations = conversations.map(conv => 
      conv.id === activeConversation.id ? updatedConversation : conv
    );
    setConversations(updatedConversations);
    
    // Simuler la réponse de l'assistant en utilisant l'API sélectionnée
    setIsThinking(true);
    
    try {
      // Appel à l'API du modèle
      const apiResponse = await generateResponse(content, selectedProvider, selectedModel);
      
      const assistantMessage = {
        id: Date.now(),
        role: 'assistant',
        content: apiResponse || `Je suis BRAIN, un assistant IA basé sur le modèle ${selectedModel}. Je ne peux pas accéder à l'API pour le moment, mais je suis là pour vous aider.`,
        timestamp: new Date().toISOString(),
        model: selectedModel
      };
      
      const conversationWithResponse = {
        ...updatedConversation,
        messages: [...updatedConversation.messages, assistantMessage],
        lastMessage: assistantMessage.content
      };
      
      setActiveConversation(conversationWithResponse);
      
      const conversationsWithResponse = updatedConversations.map(conv => 
        conv.id === activeConversation.id ? conversationWithResponse : conv
      );
      setConversations(conversationsWithResponse);
    } catch (error) {
      // En cas d'erreur, afficher un message d'erreur
      const errorMessage = {
        id: Date.now(),
        role: 'assistant',
        content: `Une erreur est survenue lors de la communication avec l'API ${selectedProvider} (${selectedModel}): ${error.message}`,
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      const conversationWithError = {
        ...updatedConversation,
        messages: [...updatedConversation.messages, errorMessage],
        lastMessage: "Erreur de communication avec l'API"
      };
      
      setActiveConversation(conversationWithError);
      
      const conversationsWithError = updatedConversations.map(conv => 
        conv.id === activeConversation.id ? conversationWithError : conv
      );
      setConversations(conversationsWithError);
    } finally {
      setIsThinking(false);
    }
  };
  
  // Créer une nouvelle conversation
  const createNewConversation = () => {
    const newConversation = {
      id: Date.now(),
      title: "Nouvelle conversation",
      date: new Date().toISOString(),
      lastMessage: "",
      messages: [{
        id: 1,
        role: "assistant",
        content: `Bonjour ! Je suis BRAIN, un assistant IA basé sur le modèle ${selectedModel}. Comment puis-je vous aider aujourd'hui ?`,
        timestamp: new Date().toISOString(),
        model: selectedModel
      }]
    };
    
    setConversations([newConversation, ...conversations]);
    setActiveConversation(newConversation);
  };
  
  // Supprimer une conversation
  const deleteConversation = (conversationId) => {
    const newConversations = conversations.filter(conv => conv.id !== conversationId);
    setConversations(newConversations);
    
    // Si la conversation active a été supprimée, sélectionner la première conversation
    if (activeConversation?.id === conversationId) {
      if (newConversations.length > 0) {
        setActiveConversation(newConversations[0]);
      } else {
        setActiveConversation(null);
      }
    }
  };

  // Renommer une conversation
  const renameConversation = (conversationId, newTitle) => {
    const updatedConversations = conversations.map(conv => 
      conv.id === conversationId ? { ...conv, title: newTitle } : conv
    );
    
    setConversations(updatedConversations);
    
    // Si la conversation active a été renommée, mettre à jour aussi activeConversation
    if (activeConversation?.id === conversationId) {
      setActiveConversation({ ...activeConversation, title: newTitle });
    }
  };
  
  return (
    <ChatContext.Provider 
      value={{
        conversations,
        activeConversation,
        isThinking,
        selectedProvider,
        selectedModel,
        setSelectedProvider,
        setSelectedModel,
        sendMessage,
        createNewConversation,
        deleteConversation,
        renameConversation,
        setActiveConversation
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};
