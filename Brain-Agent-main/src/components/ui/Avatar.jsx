import React from 'react';

const Avatar = ({ role = 'assistant' }) => {
  if (role === 'user') {
    return (
      <div className="flex-shrink-0 bg-gradient-to-br from-brand-blue to-blue-500 rounded-full h-10 w-10 flex items-center justify-center text-white font-medium shadow-lg transition-transform duration-200 hover:scale-105">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.6} stroke="currentColor" className="w-5 h-5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
        </svg>
      </div>
    );
  }
  
  // Avatar pour BRAIN avec style distinctif
  return (
    <div className="flex-shrink-0 bg-gradient-to-br from-brand-dark to-brand-blue rounded-full h-10 w-10 flex items-center justify-center text-white font-bold shadow-lg transition-transform duration-200 hover:scale-105">
      <span className="text-sm tracking-tight">B</span>
    </div>
  );
};

export default Avatar;
