"""
Run all visualization scripts
"""
import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

scripts = [
    '01_class_distribution.py',
    '02_amount_analysis.py',
    '03_time_analysis.py',
    '04_feature_correlation.py',
    '05_model_comparison.py',
    '06_summary_dashboard.py',
]

print("Generating visualizations...")
print("=" * 50)

for script in scripts:
    script_path = os.path.join(script_dir, script)
    if os.path.exists(script_path):
        print(f"\nRunning: {script}")
        result = subprocess.run([sys.executable, script_path], 
                               capture_output=True, text=True, cwd=script_dir)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Error: {result.stderr[:200]}")
    else:
        print(f"Not found: {script}")

print("\n" + "=" * 50)
print("All visualizations complete!")
