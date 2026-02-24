import re
import json

def fix_json_backslashes(s):
    """
    Escape backslashes that aren't already escaped
    Replace single backslashes with double (but not already doubled ones)
    """
    return re.sub(r'(?<!\\)\\(?![\\"])', r'\\\\', s)


def remove_json_comments(text):
    """
    Remove # comments from JSON text, preserving # inside strings.
    Tracks whether we're inside a quoted string.
    """
    result = []
    in_string = False
    escape_next = False
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        if char == '#' and not in_string:
            # Skip to end of line
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        
        result.append(char)
        i += 1
    
    return ''.join(result)


def read_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
