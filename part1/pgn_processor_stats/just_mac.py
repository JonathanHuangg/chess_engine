import json
import os
import matplotlib.pyplot as plt

def plot_mac(dir):
    files = [
        ("mac_orig.json", "1. Original"),
        ("mac_16_write_buffer_but_1MB_send.json", "2. changes but 1MB send exists"),
        ("mac_fixed_reserve_and_MESI.json", "3. Fixed Reserve & MESI"),
        ("mac_final.json", "4. Final")
    ]

    labels = []
    wall_times = []
    throughputs = []
    imbalances = []

    for file_name, label in files:
        file_path = os.path.join(dir, file_name)

        with open(file_path, 'r') as f:
            data = json.load(f)
            
        labels.append(label)
        wall_times.append(data["timing"]["total_wall_time_sec"])

        throughputs.append(data["throughput"]["board_states_per_sec"] / 1_000_000)
        imbalances.append(data["load_balance"]["load_imbalance_pct"])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Optimizations Progression', fontsize=16, fontweight='bold')

    # wall time

    ax1.plot(labels, wall_times, marker='o', markersize=8, color='#c44e52', linewidth=2.5)
    ax1.set_title('Total Wall Time (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Seconds')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    for i, txt in enumerate(wall_times):
        ax1.annotate(f"{txt:.1f}s", (i, wall_times[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # throughput
    ax2.plot(labels, throughputs, marker='s', markersize=8, color='#55a868', linewidth=2.5)
    ax2.set_title('Throughput (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Million Boards / Sec')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    for i, txt in enumerate(throughputs):
        ax2.annotate(f"{txt:.2f}M", (i, throughputs[i]), textcoords="offset points", xytext=(0,10), ha='center')
    

    # load iambalance
    bars = ax3.bar(labels, imbalances, color='#4c72b0', width=0.4)
    ax3.set_title('Thread Load Imbalance (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Imbalance %')
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.bar_label(bars, fmt='%.1f%%', padding=3)

    plt.xticks(rotation=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_img = os.path.join(dir, "mac_optimization_progression.png")
    plt.savefig(output_img, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Generated progression chart: {output_img}")

if __name__ == "__main__":
    plot_mac("part1/pgn_processor_stats")
