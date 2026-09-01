/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        risk: {
          safe: '#22c55e',
          watch: '#eab308',
          warning: '#f97316',
          critical: '#ef4444',
        },
        panel: {
          bg: '#0f1117',
          surface: '#181b24',
          border: '#2a2e3a',
          muted: '#6b7280',
          text: '#e5e7eb',
          heading: '#f9fafb',
        },
      },
    },
  },
  plugins: [],
}
