import React from 'react';

const LoadingScreen = () => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-white dark:bg-neutral-900 z-50">
      <div className="flex flex-col items-center">
        <div className="w-16 h-16 border-t-4 border-primary-600 border-solid rounded-full animate-spin"></div>
        <h3 className="mt-4 text-xl font-medium text-neutral-800 dark:text-neutral-100">Chargement...</h3>
      </div>
    </div>
  );
};

export default LoadingScreen;
