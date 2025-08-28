import React from 'react';

const ApiKeyMessage = () => {
  return (
    <div className="p-4 rounded-lg border border-yellow-300 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 text-neutral-700 dark:text-neutral-300 mb-4">
      <h3 className="text-base font-medium mb-2">Configuration des clés API</h3>
      <p className="text-sm mb-2">
        Pour utiliser les différents modèles de langage, vous devez configurer vos clés API :
      </p>
      <ol className="text-sm list-decimal pl-5 space-y-1">
        <li>Créez un fichier <code className="px-1 py-0.5 bg-neutral-200 dark:bg-neutral-800 rounded">.env.local</code> à la racine du projet</li>
        <li>Copiez le contenu du fichier <code className="px-1 py-0.5 bg-neutral-200 dark:bg-neutral-800 rounded">sample.env</code></li>
        <li>Ajoutez vos clés API pour les services que vous souhaitez utiliser</li>
        <li>Redémarrez l'application</li>
      </ol>
      <p className="text-sm mt-3">
        Les modèles non configurés resteront disponibles dans le sélecteur mais retourneront un message d'erreur en cas d'utilisation.
      </p>
    </div>
  );
};

export default ApiKeyMessage;
