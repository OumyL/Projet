import React, { useState } from 'react';
import Sidebar from './Sidebar';
import ChatContainer from './ChatContainer';

const Layout = () => {
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  
  const handleOpenSidebar = () => {
    setIsMobileOpen(true);
  };
  
  const handleCloseSidebar = () => {
    setIsMobileOpen(false);
  };

  const toggleSidebarVisibility = () => {
    setIsSidebarVisible(!isSidebarVisible);
  };

  return (
    <div className="h-screen flex overflow-hidden bg-white dark:bg-brand-dark">
      <Sidebar 
        isMobileOpen={isMobileOpen} 
        onClose={handleCloseSidebar} 
        isVisible={isSidebarVisible}
      />
      <ChatContainer 
        onOpenSidebar={handleOpenSidebar}
        onToggleSidebar={toggleSidebarVisibility}
        isSidebarVisible={isSidebarVisible}
      />
    </div>
  );
};

export default Layout;
