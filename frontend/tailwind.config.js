/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        positive: {
          light:   "#d1fae5",
          DEFAULT: "#10b981",
          dark:    "#065f46",
        },
        negative: {
          light:   "#fee2e2",
          DEFAULT: "#ef4444",
          dark:    "#7f1d1d",
        },
        neutral: {
          light:   "#fef9c3",
          DEFAULT: "#f59e0b",
          dark:    "#78350f",
        },
      },
    },
  },
  plugins: [],
};