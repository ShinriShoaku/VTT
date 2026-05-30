#!/bin/bash

# 1. Tentukan rentang tahun/bulan yang mau diisi (Format: YYYY-MM-DD)
START_DATE="2026-01-01"
END_DATE="2026-05-30"

# Konversi tanggal ke format detik (epoch) untuk looping
current=$(date -d "$START_DATE" +%s)
end=$(date -d "$END_DATE" +%s)

while [ "$current" -le "$end" ]; do
    # Format detik kembali ke tanggal ISO
    commit_date=$(date -d "@$current" +%Y-%m-%d)
    
    # 2. Bikin efek "jarang-jarang" (Probabilitas)
    # Generasi angka acak 1-10. Kalau angka yang keluar kurang dari atau sama dengan 4, baru commit.
    # Ini artinya ada peluang ~40% untuk commit di hari tersebut, sisanya (60%) bakal bolong/kosong.
    if [ $(( RANDOM % 10 + 1 )) -le 4 ]; then
        
        # 3. Jumlah commit acak per hari (misal: antara 1 sampai 4 commit)
        num_commits=$(( RANDOM % 4 + 1 ))
        
        for ((i=1; i<=num_commits; i++)); do
            # Lakukan empty commit dengan tanggal spesifik
            git commit --allow-empty --date="$commit_date 12:$(printf "%02d" $((RANDOM%60))):$(printf "%02d" $((RANDOM%60)))" -m "Update assets and core configuration" --quiet
        done
        
        echo "Mengisi tanggal $commit_date dengan $num_commits commit."
    else
        echo "Tanggal $commit_date dilewati (biar natural)."
    fi
    
    # Maju ke hari berikutnya (86400 detik = 1 hari)
    current=$((current + 86400))
done

echo "Selesai! Jangan lupa ketik 'git push origin main' (atau nama branch lu) buat update ke GitHub."
