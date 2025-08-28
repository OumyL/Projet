import React, { useContext } from 'react';
import { ChatContext } from '../../context/ChatContext';
import { ThemeContext } from '../../context/ThemeContext';
import MessageBubble from './MessageBubble';
import ThinkingIndicator from './ThinkingIndicator';
import { useChatInteraction } from '../../hooks/useChatInteraction';

const ChatMessages = () => {
  const { activeConversation, isThinking } = useContext(ChatContext);
  const { isDark } = useContext(ThemeContext);
  const { messagesEndRef } = useChatInteraction();

  // Choisir le logo en fonction du thème
  const logoSrc = isDark ? '/images/brain-logo-d.png' : '/images/brain-logo-l.png';

  if (!activeConversation) {
    return (
      <div className="flex-1 p-4 flex flex-col items-center justify-center">
        <div className="glass-card text-center max-w-md p-6 animate-fade-in">
          {/* Logo beaucoup plus grand sur l'écran d'accueil */}
          <div className="mx-auto mb-8 max-w-xs w-full">
            <img 
              src={logoSrc}
              alt="BRAIN Logo" 
              className="w-full h-auto max-h-40 object-contain mx-auto"
              onError={(e) => {
                e.target.style.display = 'none';
                // Si l'image ne charge pas, afficher le cercle avec B
                document.getElementById('fallback-circle').style.display = 'flex';
              }} 
            />
            <div 
              id="fallback-circle"
              className="w-40 h-40 rounded-full bg-gradient-to-br from-brand-blue to-blue-500 flex items-center justify-center text-white text-6xl font-bold mx-auto shadow-xl"
              style={{display: 'none'}} // Caché par défaut
            >
              B
            </div>
          </div>
          <h2 className="text-3xl font-bold text-brand-blue mb-4">Bienvenue sur BRAIN</h2>
          <p className="text-neutral-600 dark:text-brand-gray mb-6">
            Votre assistant IA personnel avec qui vous pouvez discuter de tout sujet. Posez une question pour commencer.
          </p>
          <div className="inline-block bg-gradient-to-r from-brand-blue to-blue-500 text-white px-4 py-2 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 cursor-default">
            Sélectionnez une conversation
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 px-4 md:px-6 py-4 overflow-y-auto custom-scrollbar">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Messages de la conversation */}
        {activeConversation.messages.map((message, index) => (
          <div key={message.id} className="animate-fade-in" style={{ animationDelay: `${index * 50}ms` }}>
            <MessageBubble message={message} />
          </div>
        ))}
        
        {isThinking && <ThinkingIndicator />}
        
        <div ref={messagesEndRef} className="h-4" />
      </div>
    </div>
  );
};

export default ChatMessages;
