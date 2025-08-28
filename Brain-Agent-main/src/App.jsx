import React from 'react';
import { ChatProvider } from './context/ChatContext';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/layout/Layout';
import './styles/globals.css';

function App() {
  return (
    <ThemeProvider>
      <ChatProvider>
        <div className="font-sans text-neutral-800 dark:text-brand-white bg-white dark:bg-brand-dark min-h-screen antialiased">
          <Layout />
        </div>
      </ChatProvider>
    </ThemeProvider>
  );
}

export default App;
