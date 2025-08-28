export const mockConversations = [
  {
    id: 1,
    title: "Introduction à Claude",
    date: "2023-07-12T10:30:00Z",
    lastMessage: "Comment puis-je t'aider aujourd'hui ?",
    messages: [
      {
        id: 1,
        role: "assistant",
        content: "Bonjour ! Je suis Claude, ton assistant IA. Comment puis-je t'aider aujourd'hui ?",
        timestamp: "2023-07-12T10:30:00Z"
      }
    ]
  },
  {
    id: 2,
    title: "Questions sur la programmation",
    date: "2023-07-11T14:22:00Z",
    lastMessage: "Les frameworks JavaScript les plus populaires sont React, Vue et Angular.",
    messages: [
      {
        id: 1,
        role: "user",
        content: "Quels sont les frameworks JavaScript les plus populaires en 2023 ?",
        timestamp: "2023-07-11T14:20:00Z"
      },
      {
        id: 2,
        role: "assistant",
        content: "Les frameworks JavaScript les plus populaires sont React, Vue et Angular. React est maintenu par Facebook et est particulièrement apprécié pour son approche basée sur les composants et son DOM virtuel efficace.",
        timestamp: "2023-07-11T14:22:00Z"
      }
    ]
  },
  {
    id: 3,
    title: "Planification de voyage",
    date: "2023-07-10T09:15:00Z",
    lastMessage: "Voici quelques destinations à considérer...",
    messages: [
      {
        id: 1,
        role: "user",
        content: "Je cherche des idées de destination pour des vacances en Europe cet été.",
        timestamp: "2023-07-10T09:10:00Z"
      },
      {
        id: 2,
        role: "assistant",
        content: "Voici quelques destinations à considérer en Europe pour l'été : Barcelone en Espagne, les îles grecques comme Santorin, la côte amalfitaine en Italie, ou encore Dubrovnik en Croatie. Toutes ces destinations offrent un excellent climat, de belles plages et une riche culture.",
        timestamp: "2023-07-10T09:15:00Z"
      }
    ]
  }
];
