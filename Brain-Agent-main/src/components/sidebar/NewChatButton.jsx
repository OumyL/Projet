import React, { useContext } from 'react';
import { ChatContext } from '../../context/ChatContext';

const NewChatButton = () => {
  const { createNewConversation } = useContext(ChatContext);

  return (
    <button 
      onClick={createNewConversation}
      className="flex items-center justify-center w-full py-3.5 px-4 text-sm font-medium text-white 
                bg-gradient-to-r from-brand-blue to-blue-500 hover:from-brand-blue hover:to-blue-600
                rounded-xl transition-all duration-300 shadow-md hover:shadow-lg
                transform hover:-translate-y-0.5 active:translate-y-0 active:shadow-md
                group"
    >
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" 
           className="w-5 h-5 mr-2 transition-transform duration-300 group-hover:rotate-90">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
      </svg>
      Nouvelle conversation
    </button>
  );
};

export default NewChatButton;
