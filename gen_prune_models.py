#pylint: disable-all
import subprocess
from nxmodel import gen_overall_model

generated_str = """
mnasnet_100_prune6_stem30_ceil
mnasnet_100_prune6_stem30_floor
mnasnet_100_prune5_stem30_ceil
mnasnet_100_prune5_stem30_floor
"""

names = generated_str.strip().split("\n")
for key in gen_overall_model.prune_model_dct.keys():
    if key in names:
        print("Skip already generated {}".format(key))
        continue
    print("Generate {}".format(key))
    if "192" in key:
        subprocess.check_call("python gen_elf.py --base-result-dir ./elf_results/prune_models_latencyonly/ -n {} --calib-iter 0 --mode debug --no-pretrained --input-size 192 --debug >/dev/null 2>&1".format(key), shell=True)
    else:
        subprocess.check_call("python gen_elf.py --base-result-dir ./elf_results/prune_models_latencyonly/ -n {} --calib-iter 0 --mode debug --no-pretrained --debug >/dev/null 2>&1".format(key), shell=True)
