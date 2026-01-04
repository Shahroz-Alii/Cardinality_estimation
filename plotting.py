import pandas as pd
import matplotlib.pyplot as plt

def plot_memory_impact(csv_file):
    df = pd.read_csv(csv_file)
    # Let's plot for one specific file, e.g., dracula.txt
    subset = df[df['Filename'] == 'dracula.txt']
    
    plt.figure(figsize=(10, 6))
    plt.plot(subset['Memory'], subset['HLL_Err'], marker='o', label='HLL Error %')
    plt.plot(subset['Memory'], subset['REC_Err'], marker='s', label='REC Error %')
    plt.plot(subset['Memory'], subset['Theory'], '--', label='Theoretical HLL Bound', color='red')
    
    plt.xscale('log', base=2) # Memory increases by powers of 2
    plt.xlabel('Memory (m or k)')
    plt.ylabel('Relative Error %')
    plt.title('Memory Impact on Cardinality Estimation (Dracula)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('memory_impact.png')
    plt.show()

if __name__ == "__main__":
    plot_memory_impact('results.csv')