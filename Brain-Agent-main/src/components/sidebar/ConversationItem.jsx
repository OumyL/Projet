import React, { useState, useContext, useRef, useEffect } from 'react';
import { ChatContext } from '../../context/ChatContext';
import DeleteConversationButton from './DeleteConversationButton';

const ConversationItem = ({ conversation }) => {
  const { activeConversation, setActiveConversation, renameConversation } = useContext(ChatContext);
  const [isEditing, setIsEditing] = useState(false);
  const [editedTitle, setEditedTitle] = useState(conversation.title);
  const inputRef = useRef(null);
  
  const isActive = activeConversation?.id === conversation.id;
  
  // Focus sur l'input quand on passe en mode édition
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);
  
  // Formater la date pour afficher juste "aujourd'hui" ou la date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const today = new Date();
    
    if (date.toDateString() === today.toDateString()) {
      return "Aujourd'hui";
    }
    
    return date.toLocaleDateString('fr-FR', {
      month: 'short',
      day: 'numeric'
    });
  };

  // Gestion du renommage
  const handleRename = () => {
    if (editedTitle.trim() !== '') {
      renameConversation(conversation.id, editedTitle.trim());
      setIsEditing(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleRename();
    } else if (e.key === 'Escape') {
      setEditedTitle(conversation.title);
      setIsEditing(false);
    }
  };

  // Double-clic pour éditer
  const handleClick = () => {
    if (!isEditing) {
      setActiveConversation(conversation);
    }
  };

  const handleDoubleClick = (e) => {
    e.stopPropagation();
    setIsEditing(true);
  };

  return (
    <div 
      className={`sidebar-item ${isActive ? 'sidebar-item-active' : ''} group`}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
    >
      <div className="flex-1 min-w-0">
        {isEditing ? (
          <input
            ref={inputRef}
            type="text"
            value={editedTitle}
            onChange={(e) => setEditedTitle(e.target.value)}
            onBlur={handleRename}
            onKeyDown={handleKeyDown}
            className="w-full px-1 py-0.5 text-sm bg-white dark:bg-brand-dark border border-brand-blue rounded-sm focus:outline-none focus:ring-1 focus:ring-brand-blue"
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <>
            <p className="text-sm font-medium truncate">
              {conversation.title}
            </p>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate">
              {conversation.lastMessage || "Nouvelle conversation"}
            </p>
          </>
        )}
      </div>
      
      {!isEditing && (
        <div className="flex items-center ml-2">
          <span className="text-xs text-neutral-400 dark:text-neutral-500 mr-2">
            {formatDate(conversation.date)}
          </span>
          <div className="absolute right-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <DeleteConversationButton conversationId={conversation.id} />
          </div>
        </div>
      )}
    </div>
  );
};

export default ConversationItem;
