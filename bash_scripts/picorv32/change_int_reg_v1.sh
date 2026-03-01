filename="/path/to/test.vcd"
  
if [ -f "$filename" ]; then
    sed -i 's/integer/reg/g' "$filename"
    echo "Processed $filename"
else
    echo "File $filename does not exist"
fi

echo "All files processed."
