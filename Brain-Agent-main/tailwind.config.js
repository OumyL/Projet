/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class', // Active le mode sombre basé sur les classes
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#0a1b1d',     // Couleur 1: bleu-vert très sombre
          white: '#ffffff',    // Couleur 2: blanc
          blue: '#3a97c9',     // Couleur 3: bleu clair
          gray: '#b7cad3',     // Couleur 4: gris-bleu clair
        },
        primary: {
          50: '#e6f4f9',
          100: '#cce9f3',
          200: '#99d3e8',
          300: '#66bddc',
          400: '#3aa7d1',
          500: '#3a97c9',     // Couleur principale BRAIN
          600: '#2d78a1',
          700: '#1f5a79',
          800: '#123c51',
          900: '#061e28',
          950: '#0a1b1d',     // Couleur de fond BRAIN
        },
        neutral: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
        user: {
          bubble: '#e6f4f9', // Light blue-gray for user messages
          bubbleDark: '#123c51', // Dark mode user bubble
        },
        assistant: {
          bubble: '#ffffff', // White for assistant messages
          bubbleDark: '#0f2429', // Dark mode assistant bubble - unified color
        },
        sidebar: {
          bg: '#f9fafb',
          bgDark: '#0a1b1d',
          hover: '#e6f4f9',
          hoverDark: '#0f2429',
          active: '#cce9f3',
          activeDark: '#0f2429',
          text: '#123c51',
          textDark: '#b7cad3',
        }
      },
      boxShadow: {
        'message': '0 1px 2px rgba(0, 0, 0, 0.05)',
        'messageDark': '0 1px 2px rgba(0, 0, 0, 0.25)',
        'input': '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'inputDark': '0 1px 3px rgba(0, 0, 0, 0.5), 0 1px 2px rgba(0, 0, 0, 0.3)',
      },
      animation: {
        'thinking-dot': 'thinking 1.4s infinite ease-in-out both',
      },
      keyframes: {
        thinking: {
          '0%, 80%, 100%': { transform: 'scale(0)' },
          '40%': { transform: 'scale(1.0)' },
        },
      },
    },
  },
  plugins: [],
}
