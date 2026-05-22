import json
import glob
import os 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_indiviudal_charts(dir):
    json_files = glob.glob(os.path.join(dir, "*.json"))

    if not json_files:
        print("no json file stats in {dir}")
        return 
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        threads = data.get("threads", [])
        if not threads:
            print("could not get thread data for {file_name}")
            continue 
        
        thread_ids = [f"T{t['thread_id']}" for t in threads]
        latencies = [t["wall_time_sec"] for t in threads]
        throughputs = [t["board_states"] / t["wall_time_sec"] for t in threads]

        fig = plt.figure(figsize = (14, 10))
        fig.suptitle(f'PGN Processor Statistics: {file_name}', fontsize=18, fontweight='bold', y=0.98)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6])
        fig.patch.set_facecolor('#f8f9fa')

        # plot the thread latency
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(thread_ids, latencies, color='#4c72b0', edgecolor='black')
        ax1.set_title('Thread Latency (Wall Time)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Seconds')
        ax1.bar_label(bars1, fmt='%.1f', padding=3)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.margins(y=0.2)

        # plot the thread throughput
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(thread_ids, throughputs, color='#55a868', edgecolor='black')
        ax2.set_title('Thread Throughput', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Board States / Sec')
        # set the labels in terms of thousands
        labels2 = [f'{val/1000:.0f}K' for val in throughputs]
        ax2.bar_label(bars2, labels=labels2, padding=3)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.margins(y=0.2)

        # data processed per thread
        ax3 = fig.add_subplot(gs[1, :])
        mb_processed = [t["bytes_input"] / (1024 * 1024) for t in threads]
        bars3 = ax3.bar(thread_ids, mb_processed, color='#c44e52', edgecolor='black', width=0.5)
        ax3.set_title('Data Processed per Thread', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Input Data (MB)')
        ax3.bar_label(bars3, fmt='%.0f', padding=3)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.margins(y=0.2)

        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis('off')

        work = data.get("work", {})
        tp = data.get("throughput", {})
        lb = data.get("load_balance", {})
        timing = data.get("timing", {})

        # this was stolen from gemini to make it look prettier
        summary_text = (
            f"🚀 GLOBAL WORK STATS\n"
            f"--------------------\n"
            f"Total Games:    {work.get('total_games', 0):,}\n"
            f"Total Boards:   {work.get('total_board_states', 0):,}\n"
            f"Total Data In:  {work.get('total_bytes_in', 0) / (1024**3):.2f} GB\n"
            f"Total Data Out: {work.get('total_bytes_out', 0) / (1024**3):.2f} GB\n\n"
            f"⚡ GLOBAL THROUGHPUT\n"
            f"--------------------\n"
            f"Games/sec:   {tp.get('games_per_sec', 0):,.0f}\n"
            f"Boards/sec:  {tp.get('board_states_per_sec', 0):,.0f}\n"
            f"Input MB/s:  {tp.get('input_mb_per_sec', 0):.2f}\n"
            f"Output MB/s: {tp.get('output_mb_per_sec', 0):.2f}\n"
        )
        
        load_balance_text = (
            f"⚖️  LOAD BALANCE\n"
            f"----------------\n"
            f"Imbalance Pct:      {lb.get('load_imbalance_pct', 0):.2f}%\n"
            f"Thread Time Avg:    {lb.get('thread_time_avg_sec', 0):.2f}s\n"
            f"Thread Time Min:    {lb.get('thread_time_min_sec', 0):.2f}s\n"
            f"Thread Time Max:    {lb.get('thread_time_max_sec', 0):.2f}s\n"
            f"Thread Time StdDev: {lb.get('thread_time_stddev_sec', 0):.2f}s\n\n"
            f"⏱️  TIMING\n"
            f"----------\n"
            f"Total Wall Time:   {timing.get('total_wall_time_sec', 0):.2f}s\n"
            f"File Read Time:    {timing.get('file_read_time_sec', 0):.2f}s\n"
            f"Thread Processing: {timing.get('thread_processing_time_sec', 0):.2f}s\n"
        )

        ax_text.text(0.15, 0.9, summary_text, fontsize=12, family='monospace', verticalalignment='top')
        ax_text.text(0.55, 0.9, load_balance_text, fontsize=12, family='monospace', verticalalignment='top')
        
        plt.tight_layout()

        output_img = os.path.join(dir, f"{file_name.replace('.json', '.png')}")
        plt.savefig(output_img, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Generated chart: {output_img}")

if __name__ == "__main__":
    generate_indiviudal_charts("part1/pgn_processor_stats")