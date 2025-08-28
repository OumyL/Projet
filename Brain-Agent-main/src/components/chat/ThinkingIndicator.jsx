import React from 'react';
import Avatar from '../ui/Avatar';

const ThinkingIndicator = () => {
  return (
    <div className="flex items-start gap-4 animate-fade-in">
      <Avatar role="assistant" />
      
      <div className="assistant-bubble flex items-center p-3">
        <div className="flex space-x-1.5">
          <span className="thinking-dot thinking-dot-1"></span>
          <span className="thinking-dot thinking-dot-2"></span>
          <span className="thinking-dot thinking-dot-3"></span>
        </div>
        <span className="ml-2 text-sm text-neutral-500 dark:text-brand-gray font-medium">BRAIN réfléchit...</span>
      </div>
    </div>
  );
};

export default ThinkingIndicator;
