import pandas as pd
import sys
import argparse

def analyze_patterns(csv_path):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Create masks for different conditions
        naive_true = df['Naive_Correct'] == True
        custom_true = df['Custom_Correct'] == True
        
        # Function to display full row with a header
        def display_rows(rows, header):
            if len(rows) == 0:
                print(f"\n{header}: No cases found")
            else:
                print(f"\n{header} ({len(rows)} cases):")
                for _, row in rows.iterrows():
                    print("\nInput Sentence:", row['Input_Sentence'])
                    print("Actual Last Word:", row['Actual_Last_Word'])

                    print("\nEnglish Translation:", row['English_Translation'])
                    print("English Predicted Word:", row['English_Predicted_Word'])
                    print("Naive Prediction:", row['Naive_Prediction'])

                    print("\nCustom Model Prediction:", row['Custom_Model_Prediction'])

                    print("\nNaive Correct:", row['Naive_Correct'])
                    print("Custom Correct:", row['Custom_Correct'])
                    print("-" * 80)
        
        # Display cases where only Naive is True
        naive_only = df[naive_true & ~custom_true]
        display_rows(naive_only, "Only Naive Correct")
        
        # Display cases where only Custom is True
        custom_only = df[~naive_true & custom_true]
        display_rows(custom_only, "Only Custom Correct")
        
        # Display cases where both are True
        both_true = df[naive_true & custom_true]
        display_rows(both_true, "Both Correct")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total rows: {len(df)}")
        print(f"Only Naive Correct: {len(naive_only)} cases")
        print(f"Only Custom Correct: {len(custom_only)} cases")
        print(f"Both Correct: {len(both_true)} cases")
        print(f"Both Incorrect: {len(df[~naive_true & ~custom_true])} cases")

    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_path}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze patterns in model predictions from a CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run analysis
    analyze_patterns(args.csv_file)

if __name__ == "__main__":
    main()
