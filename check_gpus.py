import subprocess
import xml.etree.ElementTree as ET
import time

def check_gpu_availability(required_space_gb=25, required_gpus=2, exlude_ids = []):
    while True:
        # Run nvidia-smi command to get GPU info in XML format
        nvidia_smi_output = subprocess.run(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE, check=True).stdout
        root = ET.fromstring(nvidia_smi_output)

        # Parse XML to find GPUs with enough free memory
        available_gpus = []
        for gpu in root.findall('gpu'):
            gpu_id = gpu.find('minor_number').text
            free_memory_mb = int(gpu.find('fb_memory_usage/free').text.replace(' MiB', ''))
            
            # Convert MB to GB for comparison
            free_memory_gb = free_memory_mb / 1024
            if free_memory_gb > required_space_gb:
                if int(gpu_id) not in exlude_ids:
                    available_gpus.append(gpu_id)

            if len(available_gpus) >= required_gpus:
                return list(map(int, available_gpus))
        
        # Pause for a short time before checking again
        time.sleep(30)

if __name__ == "__main__":
    available_gpus = check_gpu_availability(required_space_gb=10, required_gpus=2)
    print(f"GPUs with more than 25GB available space: {available_gpus}")
