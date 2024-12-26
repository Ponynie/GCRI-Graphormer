#!/bin/bash
#SBATCH --job-name=gcri_graphormer  # Job name
#SBATCH --output=job_output.log     # Output log file
#SBATCH --error=job_error.log       # Error log file
#SBATCH --partition=gpu             # Partition to use (e.g., gpu or cpu)
#SBATCH --qos=gpu20g                # QoS (Quality of Service)
#SBATCH --gres=gpu:3g.20gb:1        # Request 1 GPU of type 3g.20gb
#SBATCH --cpus-per-task=8           # Number of CPU cores
#SBATCH --mem=32G                   # Memory allocation
#SBATCH --time=06:00:00             # Maximum runtime (hh:mm:ss)

# Print some diagnostic information
echo "Job started on $(hostname) at $(date)"

# Activate the virtual environment
source venv/bin/activate

# Run your script
python src/main.py

# Print completion message
echo "Job completed at $(date)"