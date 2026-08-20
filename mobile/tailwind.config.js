/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./App.tsx', './src/**/*.{js,jsx,ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        bg: '#0D0D0D',
        card: '#1A1A1A',
        cardTop: '#22241F',
        accent: '#3DD68C',
        teal: '#1E3A3A',
      },
    },
  },
  plugins: [],
};
