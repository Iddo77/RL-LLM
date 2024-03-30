import json


def parse_json_from_substring(input_string: str):

    start_index = input_string.find('{')
    end_index = input_string.rfind('}')

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        print("Valid JSON structure not found in the input string. Returning empty JSON.")
        return {}

    try:
        json_str = input_string[start_index:end_index + 1]
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"Failed to parse extracted substring as JSON: {e}. Returning empty JSON.")
        return {}
