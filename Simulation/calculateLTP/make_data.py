import sys, os

# sys.argv[1] is the energy range you want to build
root = "/cr/tempdata01/filip/QGSJET-II/LTP/"
source = root + sys.argv[1]
ADST = root + "ADST/"
move_from = root + "temp/"
move_to = root + "temp_processed/"

# print("moving everything from source to temp")
os.system(f"mkdir {move_from} {move_to} {ADST}")
os.system(f"mv {source + '/*'} {move_from}")
while True:
    
    remaining_files = len(os.listdir(move_from))
    
    if remaining_files == 0: break
    else:
        
        print(f"remaining files: {remaining_files}")
        # os.system(f"echo mv $(ls -d {move_from}/* | head -10) {source}")
        os.system(f"mv $(ls -d {move_from}/* | head -100) {source}")
        # print(f"reading ADST files in {sys.argv[1]}")
        os.system(f"/cr/users/filip/Simulation/calculateLTP/read_ADST/AdstReader {sys.argv[1]}")
        # print(f"moving target files in {source} to {move_to}")
        os.system(f"mv {source}/* {move_to}")

os.system("/cr/users/filip/Simulation/calculateLTP/read_ADST/put_together.py")
os.system(f"rm -rf {ADST}")
os.system(f"mv {move_from}/* {source}")
os.system(f"mv {move_to}/* {source}")
os.system(f"rm -rf {move_to} {move_from}")
