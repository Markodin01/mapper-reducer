import json
from mapper import mapper
from reducer import batch_process_reducer, batch_process_filter_out


# Main processing logic
if __name__ == "__main__":
    # Load data from JSON file
    with open('azure/articles.json', 'r') as file:
        data = json.load(file)

    all_final_outputs = []
    all_percentage_reductions = []

    batch_size = 2
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_mapped_data = [mapper(record) for record in batch]
        flattened_mapped_data = [item for sublist in batch_mapped_data for item in sublist]
        batch_filtered_counts = batch_process_reducer(flattened_mapped_data)
        outputs_and_reductions = [batch_process_filter_out(batch_filtered_counts, record) for record in batch]
        
        final_outputs, percentage_reductions = zip(*outputs_and_reductions)
        all_final_outputs.extend(final_outputs)
        all_percentage_reductions.extend(percentage_reductions)

    # Flatten the results for final output
    all_final_outputs = [item for sublist in all_final_outputs for item in sublist]
    all_percentage_reductions = [item for sublist in all_percentage_reductions for item in sublist]
