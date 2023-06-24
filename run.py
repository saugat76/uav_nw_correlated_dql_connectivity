import subprocess

#Run the script for multiple levels of information exchange from level 1 to level 4
for i in range(1, 5):
    print('#######################################################')
    print('####              Running the code                 ####')
    print('#######################################################')
    p = subprocess.run(["python", "uav_env.py"])
    g = subprocess.run(["python", "main.py", "--num-episode", str(50), "--wandb-track", "True", "--learning-rate", str(2.5e-4)])

# python main.py --num-episode 350 --max-steps 1000 --wandb-track True --info-exchange-lvl 6 --nodes 512 --layers 3 --reward-func 2 --ce bf --ce-next-state False