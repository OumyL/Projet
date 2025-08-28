import { useState, useContext, useRef, useEffect } from 'react';
import { ChatContext } from '../context/ChatContext';

export const useChatInteraction = () => {
  const [message, setMessage] = useState('');
  const [rows, setRows] = useState(1);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);
  
  const { sendMessage, isThinking, activeConversation } = useContext(ChatContext);
  
  // Gérer la soumission du message
  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isThinking) {
      sendMessage(message);
      setMessage('');
      setRows(1);
    }
  };
  
  // Gérer les changements de message et ajuster la hauteur du textarea
  const handleMessageChange = (e) => {
    setMessage(e.target.value);
    
    // Calculer le nombre de lignes
    const lineHeight = 24; // hauteur d'une ligne en pixels
    const previousRows = e.target.rows;
    e.target.rows = 1; // Réinitialiser à 1 ligne
    
    const currentRows = Math.floor(e.target.scrollHeight / lineHeight);
    
    if (currentRows === previousRows) {
      e.target.rows = currentRows;
    }
    
    if (currentRows >= 1) {
      setRows(currentRows < 5 ? currentRows : 5); // Max 5 lignes
    } else {
      setRows(1);
    }
  };
  
  // Gérer les raccourcis clavier
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  // Défiler vers le bas lorsque de nouveaux messages arrivent
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeConversation?.messages]);
  
  return {
    message,
    rows,
    textareaRef,
    messagesEndRef,
    isThinking,
    handleSubmit,
    handleMessageChange,
    handleKeyDown
  };
};
