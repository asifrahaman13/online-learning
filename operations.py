# import csv

# # Function to delete columns from a CSV file
# def delete_columns(input_file, output_file, columns_to_delete):
#     with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
#         for row in reader:
#             modified_row = [item for index, item in enumerate(row) if index not in columns_to_delete]
#             writer.writerow(modified_row)

# # Example usage
# input_file = 'output1.csv'
# output_file = 'output.csv'
# columns_to_delete = [2, 2]  # Specify the column indices to delete (0-based index)

# delete_columns(input_file, output_file, columns_to_delete)


import csv

def interchange_columns(input_file, output_file, column1, column2):
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            if column1 < len(row) and column2 < len(row):
                row[column1], row[column2] = row[column2], row[column1]
            writer.writerow(row)

# Example usage
input_file = 'output.csv'
output_file = 'output1.csv'
column1 = 3  # Column index 1 (0-based)
column2 = 2 # Column index 2 (0-based)

interchange_columns(input_file, output_file, column1, column2)
