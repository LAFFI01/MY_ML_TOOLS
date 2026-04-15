import re

file_path = "/home/laffi/CODE /MY_tools/my_ml_toolkit/evaluator.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the line with "with mlflow.start_run"
mlflow_start_idx = None
mlflow_end_run_idx = None

for i, line in enumerate(lines):
    if 'with mlflow.start_run(run_name=f"AutoML' in line:
        mlflow_start_idx = i
    if 'mlflow.end_run()' in line and i > 1400:  # Ensure it's at the end
        mlflow_end_run_idx = i

if mlflow_start_idx is not None:
    print(f"Found mlflow.start_run at line {mlflow_start_idx + 1}")
    
    # Need to indent from mlflow_start_idx+1 to the return statement (before mlflow.end_run)
    # Find where to stop - should be just before the final section comment
    return_idx = None
    for i in range(mlflow_start_idx + 1, len(lines)):
        if 'return {' in lines[i] and 'summary_df' in lines[i+1]:
            return_idx = i
            break
    
    if return_idx:
        print(f"Found return statement at line {return_idx + 1}")
        
        # Indent lines from mlflow_start_idx+1 to return_idx (inclusive)
        for i in range(mlflow_start_idx + 1, return_idx + 1):
            # Only indent if the line is not empty
            if lines[i].strip():
                # Add 4 spaces at the start
                lines[i] = "        " + lines[i]
            # Else leave empty lines as-is (with just newline)
        
        # Remove or comment out mlflow.end_run() and adjust following lines
        if mlflow_end_run_idx:
            # Keep the return inside the with block by removing mlflow.end_run()
            lines[mlflow_end_run_idx] = ""  # Remove the end_run line
            # Also remove the vprint message before the return
            if mlflow_end_run_idx > 0 and 'vprint' in lines[mlflow_end_run_idx - 1]:
                lines[mlflow_end_run_idx - 1] = ""
        
        # Write back
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print("✅ Indentation complete. mlflow.end_run() removed.")
    else:
        print("❌ Could not find return statement")
else:
    print("❌ Could not find mlflow.start_run context manager")
