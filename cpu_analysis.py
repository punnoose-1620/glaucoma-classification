#!/usr/bin/env python3
"""
CPU Usage Analysis Script
Identifies causes of CPU spikes during training
"""

import psutil
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

class CPUAnalyzer:
    """Analyzes CPU usage patterns during training"""
    
    def __init__(self, log_interval=1.0):
        self.log_interval = log_interval
        self.cpu_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.gpu_history = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.monitoring = False
        
    def start_monitoring(self):
        """Start CPU monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        print("🔍 CPU monitoring started...")
        
    def stop_monitoring(self):
        """Stop CPU monitoring"""
        self.monitoring = False
        print("🛑 CPU monitoring stopped.")
        
    def _monitor_system(self):
        """Monitor system resources"""
        start_time = time.time()
        
        while self.monitoring:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                
                # Get GPU memory (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    else:
                        gpu_memory = 0
                except:
                    gpu_memory = 0
                
                # Store data
                current_time = time.time() - start_time
                self.timestamps.append(current_time)
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory.percent)
                self.gpu_history.append(gpu_memory)
                
                # Log high CPU usage
                if cpu_percent > 70:
                    print(f"⚠️  High CPU usage: {cpu_percent:.1f}% at {current_time:.1f}s")
                    
                    # Get process info
                    processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                        try:
                            if proc.info['cpu_percent'] > 5:  # Only show processes using >5% CPU
                                processes.append(proc.info)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    # Sort by CPU usage
                    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
                    
                    print("   Top CPU-consuming processes:")
                    for proc in processes[:5]:
                        print(f"     {proc['name']}: {proc['cpu_percent']:.1f}% CPU, {proc['memory_percent']:.1f}% RAM")
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                time.sleep(self.log_interval)
    
    def get_statistics(self):
        """Get statistics about CPU usage"""
        if not self.cpu_history:
            return None
            
        cpu_array = np.array(self.cpu_history)
        memory_array = np.array(self.memory_history)
        gpu_array = np.array(self.gpu_history)
        
        return {
            'cpu': {
                'mean': np.mean(cpu_array),
                'max': np.max(cpu_array),
                'min': np.min(cpu_array),
                'std': np.std(cpu_array),
                'spikes_above_70': np.sum(cpu_array > 70),
                'spikes_above_80': np.sum(cpu_array > 80),
                'spikes_above_90': np.sum(cpu_array > 90)
            },
            'memory': {
                'mean': np.mean(memory_array),
                'max': np.max(memory_array),
                'min': np.min(memory_array)
            },
            'gpu': {
                'mean': np.mean(gpu_array),
                'max': np.max(gpu_array),
                'min': np.min(gpu_array)
            },
            'duration': self.timestamps[-1] if self.timestamps else 0
        }
    
    def create_analysis_plot(self, save_path="cpu_analysis.png"):
        """Create analysis plot"""
        if not self.cpu_history:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # CPU usage over time
        axes[0].plot(list(self.timestamps), list(self.cpu_history), 'r-', linewidth=1)
        axes[0].axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='70% threshold')
        axes[0].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        axes[0].set_title('CPU Usage Over Time', fontweight='bold')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Memory usage over time
        axes[1].plot(list(self.timestamps), list(self.memory_history), 'b-', linewidth=1)
        axes[1].set_title('Memory Usage Over Time', fontweight='bold')
        axes[1].set_ylabel('Memory Usage (%)')
        axes[1].grid(True, alpha=0.3)
        
        # GPU memory usage over time
        axes[2].plot(list(self.timestamps), list(self.gpu_history), 'g-', linewidth=1)
        axes[2].set_title('GPU Memory Usage Over Time', fontweight='bold')
        axes[2].set_ylabel('GPU Memory (GB)')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Analysis plot saved to {save_path}")
    
    def print_recommendations(self):
        """Print recommendations based on analysis"""
        stats = self.get_statistics()
        if not stats:
            print("No data available for analysis")
            return
            
        print("\n" + "="*60)
        print("CPU USAGE ANALYSIS & RECOMMENDATIONS")
        print("="*60)
        
        cpu_stats = stats['cpu']
        memory_stats = stats['memory']
        gpu_stats = stats['gpu']
        
        print(f"\n📊 CPU Usage Statistics:")
        print(f"   Average CPU usage: {cpu_stats['mean']:.1f}%")
        print(f"   Peak CPU usage: {cpu_stats['max']:.1f}%")
        print(f"   CPU spikes >70%: {cpu_stats['spikes_above_70']} times")
        print(f"   CPU spikes >80%: {cpu_stats['spikes_above_80']} times")
        print(f"   CPU spikes >90%: {cpu_stats['spikes_above_90']} times")
        
        print(f"\n💾 Memory Usage Statistics:")
        print(f"   Average memory usage: {memory_stats['mean']:.1f}%")
        print(f"   Peak memory usage: {memory_stats['max']:.1f}%")
        
        print(f"\n🎮 GPU Usage Statistics:")
        print(f"   Average GPU memory: {gpu_stats['mean']:.2f} GB")
        print(f"   Peak GPU memory: {gpu_stats['max']:.2f} GB")
        
        print(f"\n⏱️  Monitoring duration: {stats['duration']:.1f} seconds")
        
        # Recommendations
        print(f"\n🔧 RECOMMENDATIONS:")
        
        if cpu_stats['mean'] > 50:
            print("   ⚠️  High average CPU usage detected!")
            print("   💡 Solutions:")
            print("     - Reduce num_workers in DataLoader (try 1-2 instead of 4)")
            print("     - Use persistent_workers=True in DataLoader")
            print("     - Simplify data augmentation transforms")
            print("     - Use non_blocking=True for tensor transfers")
            print("     - Consider using torch.compile() for model optimization")
        
        if cpu_stats['spikes_above_80'] > 10:
            print("   ⚠️  Frequent CPU spikes detected!")
            print("   💡 Solutions:")
            print("     - Increase batch size to reduce data loading frequency")
            print("     - Use prefetch_factor=2 in DataLoader")
            print("     - Implement gradient accumulation")
            print("     - Use mixed precision training")
        
        if memory_stats['mean'] > 80:
            print("   ⚠️  High memory usage detected!")
            print("   💡 Solutions:")
            print("     - Reduce batch size")
            print("     - Use gradient checkpointing")
            print("     - Implement memory-efficient data loading")
            print("     - Clear cache more frequently")
        
        if gpu_stats['mean'] < 2.0:
            print("   ⚠️  Low GPU utilization detected!")
            print("   💡 Solutions:")
            print("     - Increase batch size")
            print("     - Use larger model architecture")
            print("     - Enable mixed precision training")
            print("     - Use gradient accumulation with larger effective batch size")
        
        print(f"\n📈 Performance Summary:")
        if cpu_stats['mean'] < 30 and memory_stats['mean'] < 70 and gpu_stats['mean'] > 3.0:
            print("   ✅ Good performance balance achieved!")
        elif cpu_stats['mean'] > 60:
            print("   ⚠️  CPU bottleneck detected - optimize data loading")
        elif memory_stats['mean'] > 85:
            print("   ⚠️  Memory bottleneck detected - reduce memory usage")
        elif gpu_stats['mean'] < 2.0:
            print("   ⚠️  GPU underutilization detected - increase GPU workload")

def main():
    """Main analysis function"""
    print("="*60)
    print("CPU USAGE ANALYSIS TOOL")
    print("="*60)
    
    analyzer = CPUAnalyzer(log_interval=0.5)
    
    print("This tool will monitor your system resources during training.")
    print("Run this in a separate terminal while training to analyze CPU usage.")
    print("\nPress Ctrl+C to stop monitoring and generate analysis.")
    
    try:
        analyzer.start_monitoring()
        
        # Keep monitoring until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
        analyzer.stop_monitoring()
        
        # Generate analysis
        analyzer.create_analysis_plot()
        analyzer.print_recommendations()

if __name__ == "__main__":
    main()
