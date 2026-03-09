# Loop through files plaintext1.vcd to plaintext5000.vcd
for i in {1..3}
do
  # Define the filename
  filename="/mnt/nas/Muhib/pico_new_traces/plaintext${i}.vcd"
  
  # Check if the file exists
  if [ -f "$filename" ]; then
    # Use sed to replace 'integer' with 'reg' and save changes in the same file
    sed -i 's/integer/reg/g' "$filename"
    echo "Processed $filename"
  else
    echo "File $filename does not exist"
  fi
done

echo "All files processed."
