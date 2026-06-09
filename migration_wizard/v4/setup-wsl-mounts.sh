#!/bin/bash
# One-time setup: auto-mount all Windows drives in WSL
# Run this ONCE in a WSL terminal. It will ask for your password.
set -e
echo "=== WSL Drive Auto-Mount Setup ==="
echo "This adds your Windows drives to /etc/fstab so they auto-mount."
echo ""

DRIVES=$(cmd.exe /c "wmic logicaldisk get caption" 2>/dev/null | grep -oP '^[A-Z]:' | sed 's/://' | tr -d '\r')

for d in $DRIVES; do
    d_lower=$(echo $d | tr '[:upper:]' '[:lower:]')
    mnt="/mnt/$d_lower"
    entry="$d: $mnt drvfs defaults 0 0"
    if grep -q "$mnt" /etc/fstab 2>/dev/null; then
        echo "  Already in fstab: $entry"
    else
        echo "$entry" | sudo tee -a /etc/fstab
        echo "  Added: $entry"
    fi
    [ -d "$mnt" ] || sudo mkdir -p "$mnt"
done

echo ""
echo "Mounting all drives..."
sudo mount -a 2>/dev/null || true

echo ""
echo "Verifying..."
for d in $DRIVES; do
    d_lower=$(echo $d | tr '[:upper:]' '[:lower:]')
    mnt="/mnt/$d_lower"
    if [ -d "$mnt" ] && ls "$mnt" >/dev/null 2>&1; then
        echo "  OK: $mnt ($(ls "$mnt" | head -3 | tr '\n' ' '))"
    else
        echo "  MISSING: $mnt (drive may not be connected)"
    fi
done

echo ""
echo "=== Done! Drives will auto-mount on next WSL restart. ==="
