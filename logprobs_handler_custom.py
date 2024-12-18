import re
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class LogprobsHandler:
    def prob_to_logprob(self, prob: float) -> float:
        return np.log(prob)

    def logprob_to_prob(self, logprob: float) -> float:
        return np.exp(logprob)

    def extract_key_name(self, key: str):
        # Extract key names from the 'key_value_pair' column
        match = re.search(r'([^"]+)"\s*:', key)
        return match.group(1) if match else None

    def calculate_words_probas(self, logprobs_formatted: List[Dict]) -> List[Tuple[str, float]]:
        probas_df = pd.DataFrame({'token': [i['token'] for i in logprobs_formatted],
                                  'logprob': [i['logprob'] for i in logprobs_formatted]})

        # Combine tokens into key-value pairs
        # Assuming tokens that form key-value pairs are sequential
        key_value_pairs = []
        current_pair = []
        for idx, row in probas_df.iterrows():
            token = str(row['token'])
            if token.strip() != '' and not token.strip() in ['{', '}']:
                current_pair.append(idx)
            # Check if the token likely ends a key-value pair
            if token.endswith(',\n') or token.endswith(']\n') or token.strip().endswith(',') or token.strip().endswith(
                    '}') or token.endswith('"}') or token.endswith("'}") or token.endswith(',"') or token.endswith(
                ",'"):
                if len(current_pair) > 0:
                    key_value_pairs.append(current_pair)
                current_pair = []

        # Calculate key-value pair probabilities
        pair_probs = []
        for pair in key_value_pairs:
            pair_logprob = probas_df.loc[pair, 'logprob'].sum()
            pair_prob = self.logprob_to_prob(pair_logprob)
            pair_probs.append((''.join(probas_df.loc[pair, 'token']), pair_prob))
        return pair_probs

    def format_logprobs(self, logprobs) -> List[Dict]:
        logprobs_formatted = []
        for logprob in logprobs:
            logprob_formatted = {'token': logprob.token, 'logprob': logprob.logprob,
                                 'log_topprobs': [{'token': log_topprob.token, 'logprob': log_topprob.logprob}
                                                  for log_topprob in logprob.top_logprobs]}
            logprobs_formatted.append(logprob_formatted)
        return logprobs_formatted

    def process_logprobs(self, logprobs_formatted: List[Dict], nested_keys_dct: Dict[str, List[str]] = None):
        pair_probs = self.calculate_words_probas(logprobs_formatted)
        pair_df = pd.DataFrame(pair_probs, columns=['key_value_pair', 'agg_tokens_proba'])
        pair_df['field_name'] = pair_df['key_value_pair'].apply(self.extract_key_name)
        pair_df = pair_df[pair_df['field_name'].notna()]

        if nested_keys_dct is not None:
            for nested_key_name, nested_key_values in nested_keys_dct.items():
                nested_key_str = '|'.join(nested_key_values)
                # calculate the aggregated nested keys confidence from all the related sub-keys
                nested_rows = pair_df[pair_df['field_name'].str.contains(nested_key_str, case=False)]
                if len(nested_rows) > 0:
                    new_row = pd.DataFrame({
                        'key_value_pair': [' '.join(nested_rows['key_value_pair'])],
                        'agg_tokens_proba': [nested_rows['agg_tokens_proba'].prod()],
                        'field_name': [nested_key_name]
                    })
                    pair_df = pd.concat([pair_df, new_row], axis=0, ignore_index=True)
        pair_df['agg_tokens_proba'] = pair_df['agg_tokens_proba'].round(4)

        fields_llm_confidences = dict(pair_df.set_index('field_name')['agg_tokens_proba'].to_dict())
        return fields_llm_confidences

    def calculate_confidence_scores(self, input_data_list):
        parsed_json = {"file_name": "", "fields": []}
        current_field = None

        for i in range(len(input_data_list)):
            entry = input_data_list[i][0].strip()
            confidence = float(input_data_list[i][1])
            
            if '"file_name"' in entry:
                # Extract the file name
                parsed_json["file_name"] = entry.split(":")[1].strip(' ",')
            elif '"field_name"' in entry:
                # Append the current field before starting a new one
                if current_field:
                    parsed_json["fields"].append(current_field)
                # Start a new field
                current_field = {
                    "field_name": entry.split(":")[-1].strip(' ",'), 
                    "field_value": "", 
                    "field_confidence": confidence
                }
            elif '"field_value"' in entry and current_field:
                # Add the value to the current field
                value = entry.split(":")[-1].strip(' ",')
                current_field["field_value"] += value
                # Update the confidence to be the minimum of existing and current
                current_field["field_confidence"] = min(current_field["field_confidence"], confidence)

            # Append the field if we encounter a new "field_name" or the end of the list
            if i + 1 < len(input_data_list):
                next_entry = input_data_list[i + 1][0].strip()
                if '"field_name"' in next_entry or "]" in next_entry:
                    if current_field:
                        parsed_json["fields"].append(current_field)
                        current_field = None

        # Ensure the last field is added
        if current_field:
            parsed_json["fields"].append(current_field)

        return parsed_json

