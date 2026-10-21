#!/usr/bin/env bash
# Mirror one capsule into a Code Ocean capsule checkout: ./mirror.sh ingest [/root/capsule]
# Copies each top-level entry of capsules/<name>/code over its namesake in <checkout>/code, plus
# .codeocean/environment.json, .codeocean/resources.json and environment/postInstall. Everything
# else in the checkout (data assets, secrets, the generated Dockerfile, reference clones) is left alone.
set -euo pipefail
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$1"
dest="${2:-/root/capsule}"
[[ -d "$src/code" ]] || { echo "no capsule at $src" >&2; exit 1; }

mkdir -p "$dest/code" "$dest/.codeocean" "$dest/environment"
for entry in "$src"/code/* "$src"/code/.[!.]*; do
    [[ -e "$entry" ]] || continue
    rm -rf "${dest:?}/code/$(basename "$entry")"
    cp -a "$entry" "$dest/code/"
done
for file in .codeocean/environment.json .codeocean/resources.json environment/postInstall; do
    [[ -f "$src/$file" ]] && cp -a "$src/$file" "$dest/$file"
done
find "$dest/code" -path "$dest/code/DataScales" -prune -o \( -name __pycache__ -o -name .pytest_cache \) -type d -print0 | xargs -0 rm -rf
echo "mirrored $1 -> $dest"
