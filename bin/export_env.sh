#!/usr/bin/env bash
# Export all variables from .env file in the project root.
# Usage: source bin/export_env.sh


ENV_FILE="$(pwd)/.env"

if [[ -f "$ENV_FILE" ]]; then
    echo "Exporting environment variables from $ENV_FILE..."
    line_num=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line_num=$((line_num + 1))
        
        # Skip empty lines and lines with only whitespace
        [[ -z "${line// /}" ]] && continue
        [[ -z "${line//$'\t'/}" ]] && continue
        
        # Skip comments
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Skip lines without '='
        if [[ ! "$line" =~ = ]]; then
            echo "Warning: Skipping malformed line $line_num: '$line' (missing '=')"
            continue
        fi
        
        # Extract key (everything before the first '=')
        key="${line%%=*}"
        # Extract value (everything after the first '=')
        value="${line#*=}"
        
        # Remove leading/trailing whitespace from key
        key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        
        # Skip if key is empty
        if [[ -z "$key" ]]; then
            echo "Warning: Skipping line $line_num: empty key"
            continue
        fi
        
        # Skip if key contains invalid characters (only allow alphanumeric and underscore)
        if [[ ! "$key" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
            echo "Warning: Skipping line $line_num: invalid key '$key'"
            continue
        fi
        
        # Remove surrounding quotes from value if present
        if [[ "$value" =~ ^\"(.*)\"$ ]]; then
            value="${BASH_REMATCH[1]}"
        elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi
        
        echo "Exporting $key"
        # Export the variable
        export "$key"="$value"
    done < "$ENV_FILE"
    echo "Environment variables exported from .env."

else 
    echo "No .env file found at $ENV_FILE. Please create one with the necessary environment variables."
fi

