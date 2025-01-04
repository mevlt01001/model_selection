import time
import os


log_dir = 'log'
log_file = 'log.txt'

def log(where, message, log_file=log_file):
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, log_file)    

    # Write the log message
    with open(log_file, 'a') as f:
        dots_count = 25 - len(where) - 2  
        dots = "." * dots_count
        f.write(f"{time.ctime()}: {where}{dots}: {message}\n")