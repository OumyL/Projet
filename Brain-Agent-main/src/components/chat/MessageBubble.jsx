import React from 'react';
import Avatar from '../ui/Avatar';

const MessageBubble = ({ message }) => {
  const { role, content, model, isError } = message;
  const isUser = role === 'user';

  return (
    <div className={`flex items-start gap-4 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && <Avatar role="assistant" />}
      
      <div className={`
        ${isUser ? 'user-bubble' : 'assistant-bubble'} 
        ${isError ? 'border-red-300 dark:border-red-700' : ''}
        transform transition-all ease-out
      `}>
        <div className="whitespace-pre-wrap prose prose-sm dark:prose-invert max-w-none">
          {content}
        </div>
        
        {/* Afficher le modèle utilisé pour la réponse uniquement en cas d'erreur */}
        {!isUser && isError && model && (
          <div className="mt-3 text-xs text-red-500 dark:text-red-400 border-t border-neutral-200 dark:border-neutral-700 pt-2">
            Modèle: {model}
          </div>
        )}
      </div>
      
      {isUser && <Avatar role="user" />}
    </div>
  );
};

export default MessageBubble;
