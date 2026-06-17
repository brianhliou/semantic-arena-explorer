#!/bin/sh
# Assemble the static explorer into public/ for Vercel deployment.
set -e
rm -rf public
cp -r explorer public
cp paper/one-clue-reference.pdf public/one-clue-reference.pdf
echo "built public/ from explorer/ (+ paper PDF)"
