#!/bin/bash

# Check image format vs extension mismatch
check_image_mismatch() {
    local dir="${1:-.}"  # Use current directory if none specified
    
    # Supported image formats
    declare -A format_map=(
        ["jpeg"]="jpg jpeg"
        ["png"]="png"
        ["gif"]="gif"
        ["bmp"]="bmp"
        ["tiff"]="tif tiff"
        ["webp"]="webp"
    )

    find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \
        -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \
        -o -iname "*.webp" \) -print0 | while IFS= read -r -d $'\0' file; do
        
        # Get actual file type
        mime_type=$(file -b --mime-type "$file")
        file_type=${mime_type#image/}
	actual_format=${file_type}
        
        # Normalize file type
        case "$file_type" in
            "x-ms-bmp") file_type="bmp" ;;
            "vnd.microsoft.icon") file_type="ico" ;;
        esac

        # Get file extension
        extension=$(basename "$file" | awk -F . '{print tolower($NF)}')
	expected_format=${format_map[$extension]}
        
        # Check for mismatch
        if [[ -n "${format_map[$file_type]}" ]]; then
            if ! grep -qw "$extension" <<< "${format_map[$file_type]}"; then
                echo "MISMATCH: $file (Extension: .$extension, Actual: $file_type)"
		echo -n -e "Fixing: $file (${actual_format} -> ${expected_format})" && echo ", debug:[${file%.*}.$extension]"
		convert "$file" "$file"
		#exit 0;
            fi
        else
            echo "UNKNOWN: $file - MIME type: $mime_type"
        fi
    done
}

# Run with specified directory or current directory
check_image_mismatch "$@"
