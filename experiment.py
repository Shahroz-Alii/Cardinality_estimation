import math
import os
import csv
import numpy as np
from algorithms import HyperLogLog, Recordinality, KMV
from data_utils import get_words_from_file, generate_zipf_stream

def get_true_n_from_dat(filename):
    dat_path = filename.replace(".txt", ".dat")
    if os.path.exists(dat_path):
        with open(dat_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    return None

def run_comprehensive_study(filename, results_list):
    if not os.path.exists(filename): return
    words = get_words_from_file(filename)
    true_n = get_true_n_from_dat(filename) or len(set(words))
    
    print(f"\nDATASET: {filename} (True n: {true_n})")
    print(f"{'Mem':<5} | {'HLL Est':<9} | {'REC Est':<9} | {'KMV Est':<9} | {'HLL %':<7} | {'REC %':<7} | {'KMV %':<7} | {'Theory'}")
    print("-" * 90)
    
    for m in [64, 256, 1024, 4096]:
        h_res, r_res, k_res = [], [], []
        for _ in range(10): # 10 trials to ensure statistical reliability
            hll, rec, kmv = HyperLogLog(int(math.log2(m))), Recordinality(m), KMV(m)
            for w in words:
                hll.add(w); rec.add(w); kmv.add(w)
            h_res.append(hll.estimate()); r_res.append(rec.estimate()); k_res.append(kmv.estimate())
        
        avg_h, avg_r, avg_k = np.mean(h_res), np.mean(r_res), np.mean(k_res)
        err_h = abs(avg_h - true_n) / true_n * 100
        err_r = abs(avg_r - true_n) / true_n * 100
        err_k = abs(avg_k - true_n) / true_n * 100
        theory_hll = (1.04 / math.sqrt(m)) * 100
        
        print(f"{m:<5} | {avg_h:<9.0f} | {avg_r:<9.0f} | {avg_k:<9.0f} | "
              f"{err_h:<7.1f}% | {err_r:<7.1f}% | {err_k:<7.1f}% | {theory_hll:.2f}%")
        
        # Save to list for CSV
        results_list.append(["Part1_RealData", filename, m, true_n, avg_h, avg_r, avg_k, err_h, err_r, err_k, theory_hll, "N/A"])

def run_synthetic_study(results_list):
    print(f"\nPART 2: SYNTHETIC ZIPFIAN TEST (n=5000, N=100000, m=1024)")
    print(f"{'Alpha':<10} | {'HLL Error':<10} | {'REC Error':<10}")
    print("-" * 35)
    for a in [0.0, 0.5, 1.1, 2.0]:
        stream = generate_zipf_stream(5000, 100000, a)
        true_s = len(np.unique(stream))
        hll, rec = HyperLogLog(10), Recordinality(1024)
        for x in stream: 
            hll.add(x); rec.add(x)
        
        err_h = abs(hll.estimate()-true_s)/true_s * 100
        err_r = abs(rec.estimate()-true_s)/true_s * 100
        
        print(f"{a:<10} | {err_h:<10.2f}% | {err_r:.2f}%")
        # Save to list for CSV
        results_list.append(["Part2_Synthetic", "Zipf_Stream", 1024, true_s, hll.estimate(), rec.estimate(), 0, err_h, err_r, 0, 3.25, a])

if __name__ == "__main__":
    my_files = ["dracula.txt", "iliad.txt", "mare-balena.txt", "midsummer-nights-dream.txt", "quijote.txt", "valley-fear.txt", "war-peace.txt"]
    all_results = []
    
    # Run Part 1
    for book in my_files:
        run_comprehensive_study(book, all_results)
    
    # Run Part 2
    run_synthetic_study(all_results)
        
    # Write to CSV
    keys = ["Study_Type", "Filename", "Memory", "True_n", "HLL_Est", "REC_Est", "KMV_Est", "HLL_Err", "REC_Err", "KMV_Err", "Theory", "Alpha_Skew"]
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(all_results)
    print("\n[✔] All results (Real + Synthetic) saved to results.csv")