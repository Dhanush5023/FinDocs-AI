/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        fintech: {
          dark: '#0B0F19',
          card: '#151D2F',
          border: '#232E48',
          accent: '#3B82F6',
          teal: '#00F2FE',
          emerald: '#10B981'
        }
      }
    },
  },
  plugins: [],
}