import json
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Try to read the dictionary file
    with open(os.path.join(script_dir, 'words_dictionary.json'), 'r') as f:
        dictionary = json.load(f)
    
    # Filter out words less than 3 characters
    filtered_dict = {word: 1 for word in dictionary.keys() if len(word) >= 3}
    
    # Save the filtered dictionary back
    with open(os.path.join(script_dir, 'filtered_words_dictionary.json'), 'w') as f:
        json.dump(filtered_dict, f, indent=2)
    
    # Print some statistics
    print(f"Original dictionary size: {len(dictionary)} words")
    print(f"Filtered dictionary size: {len(filtered_dict)} words")
    print(f"Removed {len(dictionary) - len(filtered_dict)} words")
    
except FileNotFoundError:
    print("Error: words_dictionary.json not found in the same directory as this script")
except json.JSONDecodeError:
    print("Error: words_dictionary.json is not a valid JSON file")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}") 