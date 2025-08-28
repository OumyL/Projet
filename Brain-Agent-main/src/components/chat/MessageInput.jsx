import React from 'react';
import { useChatInteraction } from '../../hooks/useChatInteraction';

const MessageInput = () => {
  const { 
    message, 
    rows, 
    textareaRef, 
    isThinking,
    handleSubmit, 
    handleMessageChange, 
    handleKeyDown 
  } = useChatInteraction();

  return (
    <form 
      onSubmit={handleSubmit}
      className="flex items-end border-t border-neutral-200/70 dark:border-brand-blue/10 bg-white/70 dark:bg-brand-dark/70 backdrop-blur-sm p-4 sticky bottom-0 z-10"
    >
      <div className="relative flex-1 mx-auto max-w-4xl w-full">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={handleMessageChange}
          onKeyDown={handleKeyDown}
          placeholder="Envoyer un message à BRAIN..."
          rows={rows}
          disabled={isThinking}
          className="chat-input min-h-[56px] max-h-[200px] pr-12 dark:bg-brand-dark/80 dark:border-brand-blue/30 dark:focus:ring-brand-blue/60 placeholder-neutral-400 dark:placeholder-neutral-500"
        />
        <button
          type="submit"
          disabled={!message.trim() || isThinking}
          className={`absolute right-3 bottom-3 p-2 rounded-lg ${
            message.trim() && !isThinking 
            ? 'text-brand-blue bg-brand-blue/10 hover:bg-brand-blue/20 dark:bg-brand-blue/20 dark:hover:bg-brand-blue/30 transition-colors' 
            : 'text-neutral-300 dark:text-neutral-600'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
          </svg>
        </button>
      </div>
    </form>
  );
};

export default MessageInput;
