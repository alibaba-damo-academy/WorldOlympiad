#!/bin/bash
# Robust resumable download of a HF model repo through hf-mirror (curl -C - in a retry loop).
#   bash wo/models/fetch_hf.sh REPO DEST_DIR
# Large files hosted on xet storage break mid-stream; curl resume + retry gets them through.
set -u
REPO=$1; DEST=$2; BASE=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$DEST"
files=$(curl -s -m 60 "$BASE/api/models/$REPO/tree/main" | python3 -c "
import json,sys
for f in json.load(sys.stdin):
    if f.get('type')=='file': print(f['path'], f.get('size',0))")
echo "$files" | while read -r path size; do
  [ -z "$path" ] && continue
  out="$DEST/$path"; mkdir -p "$(dirname "$out")"
  for attempt in $(seq 1 200); do
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" = "$size" ] && break
    curl -sSL -C - --retry 5 --retry-delay 3 --connect-timeout 30 --speed-time 60 --speed-limit 10000 \
      -o "$out" "$BASE/$REPO/resolve/main/$path" || true
  done
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  echo "$path $have/$size $([ "$have" = "$size" ] && echo OK || echo INCOMPLETE)"
done
echo FETCHDONE
