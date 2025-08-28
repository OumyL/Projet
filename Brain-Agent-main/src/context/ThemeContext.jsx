import React, { createContext, useState, useEffect } from 'react';

export const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  // Vérifier si le thème sombre est stocké dans localStorage ou si l'utilisateur préfère le thème sombre
  const getInitialTheme = () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme === 'dark';
    } else {
      // Vérifier les préférences du système
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
  };

  const [isDark, setIsDark] = useState(getInitialTheme);

  // Mettre à jour la classe sur l'élément HTML
  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  const toggleTheme = () => {
    setIsDark(!isDark);
  };

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
