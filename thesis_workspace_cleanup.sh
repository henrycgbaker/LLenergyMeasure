#!/bin/bash

# ========================
# Thesis Cleanup & Report
# ========================

THESIS_DIR="/home/228755@hertie-school.lan/thesis"
LOGFILE="$THESIS_DIR/cleanup_report_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "==== Thesis Disk Usage Report ($(date)) ====" | tee "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# 1. Top folders by size
echo "🔍 Top folders (>100MB):" | tee -a "$LOGFILE"
du -h --max-depth=2 "$THESIS_DIR" 2>/dev/null | grep -E '^[0-9\.]+G|[1-9][0-9]{2}M' | sort -h | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# 2. Top 20 files by size
echo "📁 Top 20 files by size:" | tee -a "$LOGFILE"
find "$THESIS_DIR" -type f -exec du -h {} + 2>/dev/null | sort -rh | head -n 20 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# 3. Files to delete
echo "🗑️ Candidates for deletion:" | tee -a "$LOGFILE"
find "$THESIS_DIR" \( -name "*.pyc" -o -name "*.log" -o -name ".DS_Store" \) -type f -size +10M | tee -a "$LOGFILE"
find "$THESIS_DIR" -name ".ipynb_checkpoints" -type d | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# 4. Actual cleanup
echo "🚨 Cleaning up *.pyc, *.log, .ipynb_checkpoints, __pycache__, .DS_Store" | tee -a "$LOGFILE"
find "$THESIS_DIR" -name "*.pyc" -delete
find "$THESIS_DIR" -name "*.log" -delete
find "$THESIS_DIR" -name ".DS_Store" -delete
find "$THESIS_DIR" -name "__pycache__" -type d -exec rm -rf {} +
find "$THESIS_DIR" -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
echo "✅ File-type cleanup complete." | tee -a "$LOGFILE"

# 5. Clean up folders: wandb/, logs/, outputs/
echo "🧹 Cleaning folders: wandb/, logs/, outputs/" | tee -a "$LOGFILE"
for dir in "wandb" "logs" "outputs"; do
    if [ -d "$THESIS_DIR/$dir" ]; then
        echo " - Removing $dir/" | tee -a "$LOGFILE"
        rm -rf "$THESIS_DIR/$dir"
    else
        echo " - $dir/ not found, skipping." | tee -a "$LOGFILE"
    fi
done

echo "✅ Cleanup complete." | tee -a "$LOGFILE"

# Optional: compress log if large
gzip "$LOGFILE" 2>/dev/null
