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

    all_words = []
    for word_len in range(3, 16):
        try:
            print(f"Reading {word_len} wordlist")
            with open(os.path.join(script_dir, str(word_len) + '-letter-words.json')) as f:
                raw_word_list = json.load(f)
                processed_words = [item["word"] for item in raw_word_list]
                all_words.extend(processed_words)
        except FileNotFoundError:
            print(f"Error: {word_len}-letter-words.json not found in the same directory as this script")
        except json.JSONDecodeError:
            print(f"Error: {word_len}-letter-words.json is not a valid JSON file")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}") 
    
    filtered_dict = { word.lower(): 1 for word in all_words if not is_invalid_word(word)}

    # Save the filtered dictionary back
    with open(os.path.join(script_dir, 'filtered_words_dictionary.json'), 'w') as f:
        json.dump(filtered_dict, f, indent=2)

if __name__ == "__main__":
    main()