import json
import os

def is_invalid_word(word):

    if len(word) < 3:
        return True

    # words with same ajacent letters are invalid
    if any(word[i] == word[i + 1] for i in range(len(word) - 1)):
        return True
    
    if any(word[i] not in 'abcdefghijklmnopqrstuvwxyz' for i in range(len(word))):
        return True

    return False


# Get the directory of this script
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Try to read the dictionary file
        with open(os.path.join(script_dir, 'raw_words_dictionary.json'), 'r') as f:
            dictionary = json.load(f)
        
        # Filter out words less than 3 characters
        filtered_dict = {word.lower(): 1 for word in dictionary.keys() if not is_invalid_word(word)}

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

if __name__ == "__main__":
    main()