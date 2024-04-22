import json

import pandas as pd


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


def escape_brackets(prompt_text: str, variables: list[str]) -> str:
    # escape brackets in json, otherwise validation of langchain will fail
    prompt_text = prompt_text.replace('{', '{{')
    prompt_text = prompt_text.replace('}', '}}')
    for var in variables:
        prompt_text = prompt_text.replace('{{' + var + '}}', '{' + var + '}')
    return prompt_text


def trim_list(lst: list, max_size=10):
    while len(lst) > max_size:
        lst.pop(0)


def calculate_column_average(file_path, column_index):
    data = pd.read_csv(file_path)
    column = data.iloc[:, column_index]
    return column.mean()
