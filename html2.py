import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

def extract_accuracies_from_html(html_path):
    """Extract accuracy data from Plotly-generated HTML file."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        
    # Find the script tag containing the Plotly data
    script_tags = soup.find_all('script', type='text/javascript')
    for script in script_tags:
        if 'Plotly.newPlot' in script.text:
            # Extract the y-values (accuracies) for the smoothed curve
            match = re.search(r'y:\s*\[([^\]]+)\]', script.text)
            if match:
                accuracies_str = match.group(1)
                accuracies = [float(x) for x in accuracies_str.split(',')]
                return accuracies
    return None

def extract_method_name(filename):
    """Extract method name from filename."""
    method_names = ['fedavg', 'fedgta', 'fedtad', 'fgssl', 'fedprox', 'adafgl', 'lap']
    for name in method_names:
        if name in filename:
            return name
    return None

def generate_consolidated_csv(curves_dir, output_csv='./log/mychem.csv'):
    """
    Generate a consolidated CSV file from multiple HTML accuracy curves.
    
    Args:
        curves_dir (str): Root directory containing dataset subdirectories with HTML files
        output_csv (str): Path to save the output CSV file
    """
    all_data = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(curves_dir):
        # Process each dataset directory (like 'CS')
        if root != curves_dir:  # Skip the root directory
            dataset_name = os.path.basename(root)
            
            # Group files by method and run
            method_files = {}
            
            for file in files:
                if file.endswith('.html'):
                    method_name = extract_method_name(file)
                    if method_name:
                        # Extract run number (default to 0 for files without _X)
                        run_match = re.search(r'_(\d+)\.html$', file)
                        run_num = int(run_match.group(1)) if run_match else 0
                        
                        if method_name not in method_files:
                            method_files[method_name] = []
                        method_files[method_name].append((run_num, file))
            
            # Process each method's files
            for method_name, files_info in method_files.items():
                # Sort files by run number
                files_info.sort(key=lambda x: x[0])
                sorted_files = [f[1] for f in files_info]
                
                # Process each run
                for run_idx, file in enumerate(sorted_files, start=1):
                    html_path = os.path.join(root, file)
                    accuracies = extract_accuracies_from_html(html_path)
                    
                    if accuracies:
                        for round_num, accuracy in enumerate(accuracies, start=1):
                            all_data.append({
                                'Experiment Name': method_name,
                                'Dataset': dataset_name,
                                'Run ID': run_idx,
                                'Round': round_num,
                                'Mean Acc': accuracy
                            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    
    # Standardize method names (if needed)
    df['Experiment Name'] = df['Experiment Name'].replace({
        'fedavg': 'FedAvg',
        'fedprox': 'FedProx',
        # Add other mappings as needed
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Consolidated CSV saved to {output_csv}")
    
    return df

if __name__ == "__main__":
    # Example usage:
    curves_directory = "./curves"  # Root directory containing dataset subdirectories
    output_csv_path = "./log/mychem.csv"
    
    # Generate the consolidated CSV
    df = generate_consolidated_csv(curves_directory, output_csv_path)
    
    # Print summary
    if not df.empty:
        print("\nSummary of generated CSV:")
        print(f"Total datasets: {df['Dataset'].nunique()}")
        print(f"Total methods: {df['Experiment Name'].nunique()}")
        print(f"Total runs: {df.groupby(['Dataset', 'Experiment Name'])['Run ID'].nunique().sum()}")
        print(f"Total rows: {len(df)}")
        print("\nMethods and their run counts per dataset:")
        print(df.groupby(['Dataset', 'Experiment Name'])['Run ID'].nunique())