import React, { useContext } from 'react';
import { ChatContext } from '../../context/ChatContext';
import { ThemeContext } from '../../context/ThemeContext';
import NewChatButton from '../sidebar/NewChatButton';
import ConversationItem from '../sidebar/ConversationItem';
import ThemeToggle from '../ui/ThemeToggle';

const Sidebar = ({ isMobileOpen, onClose, isVisible }) => {
  const { conversations } = useContext(ChatContext);
  const { isDark } = useContext(ThemeContext);

  // Choisir le logo en fonction du thème
  const logoSrc = isDark ? '/images/brain-logo-d.png' : '/images/brain-logo-l.png';

  // Classes CSS pour le comportement responsive et la visibilité
  const sidebarClasses = `
    bg-white dark:bg-brand-dark
    border-r border-neutral-200/70 dark:border-brand-blue/10 flex flex-col h-full
    md:relative md:min-w-[0px] md:transition-all md:duration-300 md:ease-in-out
    ${isVisible ? 'md:w-80 md:opacity-100' : 'md:w-0 md:opacity-0 md:overflow-hidden'}
    ${isMobileOpen 
      ? 'fixed inset-0 z-40 w-full backdrop-blur-lg' 
      : 'hidden md:flex'
    }
  `;

  return (
    <aside className={sidebarClasses}>
      {/* En-tête du sidebar */}
      <div className="p-4 pb-4 border-b border-neutral-200/70 dark:border-brand-blue/10">
        {/* Logo BRAIN */}
        <div className="relative w-full border-2 border-brand-blue/30 dark:border-brand-blue/20 rounded-lg p-3 bg-white/80 dark:bg-brand-dark/80 shadow-sm">
          <img 
            src={logoSrc}
            alt="BRAIN Logo" 
            className="w-full h-auto max-h-16 object-contain"
            onError={(e) => {
              e.target.style.display = 'none';
              document.getElementById('text-logo').style.display = 'block';
            }} 
          />
          <h1 
            id="text-logo" 
            className="text-3xl font-bold bg-gradient-to-r from-brand-blue to-blue-500 bg-clip-text text-transparent text-center"
            style={{display: 'none'}}
          >
            BRAIN
          </h1>
          
          {/* Bouton mode clair/sombre */}
          <div className="absolute top-2 right-2">
            <ThemeToggle />
          </div>
          
          {/* Bouton fermeture mobile */}
          {isMobileOpen && (
            <button 
              onClick={onClose}
              className="md:hidden absolute top-2 left-2 p-1.5 rounded-md bg-white/80 dark:bg-brand-dark/80 text-neutral-500 hover:text-neutral-700 dark:text-neutral-400 dark:hover:text-neutral-200"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
      
      {/* Bouton nouvelle conversation */}
      <div className="p-4">
        <NewChatButton />
      </div>
      
      {/* Titre de la section */}
      <div className="px-4 mb-2">
        <h2 className="text-xs text-neutral-500 dark:text-brand-gray font-medium uppercase tracking-wider">Conversations récentes</h2>
      </div>
      
      {/* Liste des conversations */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1 custom-scrollbar">
        {conversations.length > 0 ? (
          conversations.map(conversation => (
            <ConversationItem 
              key={conversation.id} 
              conversation={conversation}
            />
          ))
        ) : (
          <div className="text-center p-4 text-neutral-500 dark:text-neutral-400 text-sm italic">
            Aucune conversation. <br />
            Cliquez sur "Nouvelle conversation" pour commencer.
          </div>
        )}
      </div>
      
      {/* Pied de page */}
      <div className="p-4 border-t border-neutral-200/70 dark:border-brand-blue/10 text-xs text-center">
        <p className="bg-gradient-to-r from-brand-blue to-blue-500 bg-clip-text text-transparent font-semibold">
          BRAIN by Brain Gen Technology
        </p>
      </div>
    </aside>
  );
};

export default Sidebar;
