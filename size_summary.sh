#!/bin/bash

echo "🔍 REPOSITORY SIZE ANALYSIS"
echo "=========================="
echo

# Total size
echo "📊 TOTAL REPOSITORY SIZE:"
du -sh . 2>/dev/null | awk '{print "   " $1}'
echo

# Count files
echo "📁 FILE COUNT:"
find . -type f | wc -l | awk '{print "   " $1 " files"}'
echo

# Directory sizes
echo "📂 DIRECTORY BREAKDOWN:"
echo "   Code files (.py, .ipynb, .sh, .md):"
find . -name "*.py" -o -name "*.ipynb" -o -name "*.sh" -o -name "*.md" | xargs du -ch 2>/dev/null | tail -1 | awk '{print "     " $1}'

echo "   Data directories:"
du -sh data/ cache/ train_test/ benchmarks/ 2>/dev/null | awk '{print "     " $2 ": " $1}'

echo "   Models & Checkpoints:"
du -sh models/ checkpoints/ 2>/dev/null | awk '{print "     " $2 ": " $1}'

echo "   Output & Results:"
du -sh output/ results/ logs/ experiments/ 2>/dev/null | awk '{print "     " $2 ": " $1}'

echo "   Configs:"
du -sh configs/ slurm/ 2>/dev/null | awk '{print "     " $2 ": " $1}'

echo "   Git:"
du -sh .git/ 2>/dev/null | awk '{print "     " $1}'
echo

# Largest files
echo "📈 LARGEST FILES (top 10):"
find . -type f -exec du -h {} + 2>/dev/null | sort -hr | head -10 | awk '{print "   " $1 " - " $2}' 