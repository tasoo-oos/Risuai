set -e

export VITE_RISU_LEGAL_CONFIGURED=TRUE

npm install -g pnpm
pnpm install
pnpm run build
pnpm run runserver
