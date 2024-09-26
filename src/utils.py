import csv

def save_predictions_to_csv(predictions, file_name="predicted_values.csv"):
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(predictions)
    print(f"CSV file '{file_name}' created successfully.")
