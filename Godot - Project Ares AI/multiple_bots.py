import subprocess

for port in range(12345, 12347):  # Adjust range for your number of bots
    subprocess.Popen(['python', 'usemodel.py', '--mode', 'auto', '--port', str(port)])