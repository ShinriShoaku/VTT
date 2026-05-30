#!/bin/bash

# 1. Rentang tanggal diubah penuh ke tahun 2025
START_DATE="2025-01-01"
END_DATE="2025-12-31"

current=$(date -d "$START_DATE" +%s)
end=$(date -d "$END_DATE" +%s)

while [ "$current" -le "$end" ]; do
    commit_date=$(date -d "@$current" +%Y-%m-%d)
    
    # Peluang commit 40% biar jarang-jarang dan natural
    if [ $(( RANDOM % 10 + 1 )) -le 4 ]; then
        
        # Jumlah commit acak per hari (1 sampai 4 commit)
        num_commits=$(( RANDOM % 4 + 1 ))
        
        for ((i=1; i<=num_commits; i++)); do
            # Mengisi jam, menit, dan detik secara acak agar terlihat realistik
            git commit --allow-empty --date="$commit_date $((RANDOM%12+9)):$(printf "%02d" $((RANDOM%60))):$(printf "%02d" $((RANDOM%60)))" -m "Refactor core modules and optimize performance" --quiet
        done
        
        echo "Mengisi tanggal $commit_date dengan $num_commits commit."
    else
        echo "Tanggal $commit_date dilewati."
    fi
    
    current=$((current + 86400))
done

echo "Selesai memproses tahun 2025!"
echo "Silakan push dengan perintah: git push origin main"