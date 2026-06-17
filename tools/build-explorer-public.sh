#!/bin/sh
# Assemble the static explorer into public/ for Vercel deployment.
set -e
rm -rf public
cp -r explorer public
echo "built public/ from explorer/"
