import os
import time
import torch
import argparse
from typing import Optional
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class PipelineStage(nn.Module):
    """
    One stage of our pipeline (a simple feed-forward block).

    In a real model like GPT, each stage might be several transformer layers.
    """

    def __init__(self, input_size: int, output_size: int, stage_id: int):
        super().__init__()
        self.stage_id = stage_id
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

def visualize_pipeline():
    """Print a visualization of pipeline parallelism."""
    print("""
    ═══════════════════════════════════════════════════════════════════════
    PIPELINE PARALLELISM VISUALIZATION
    ═══════════════════════════════════════════════════════════════════════

    The model is split across GPUs/processes:

    Full Model:     [Embed] → [Layer 0-3] → [Layer 4-7] → [Layer 8-11] → [Head]
                        ↓           ↓            ↓             ↓           ↓
    Pipeline:       Stage 0     Stage 1      Stage 2       Stage 3     Stage 4

    Data flows through stages via send/recv:

    Time →
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  Stage 0:  [MB0 Fwd]─────►[MB1 Fwd]─────►[MB2 Fwd]─────►[MB3 Fwd]     │
    │                 │              │              │              │         │
    │                 ▼              ▼              ▼              ▼         │
    │  Stage 1:      [MB0 Fwd]─────►[MB1 Fwd]─────►[MB2 Fwd]─────►[MB3 Fwd] │
    │                     │              │              │              │     │
    │                     ▼              ▼              ▼              ▼     │
    │  Stage 2:          [MB0 Fwd]─────►[MB1 Fwd]─────►[MB2 Fwd]─────►...   │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

    MB = Microbatch, Fwd = Forward pass

    Key insight: While Stage 2 processes MB0, Stage 1 processes MB1,
    and Stage 0 processes MB2. The pipeline is "full" of work!

    ═══════════════════════════════════════════════════════════════════════
    """)

def pipeline_worker(rank:int,
                    world_size:int,
                    batch_size:int,
                    hidden_size:int,
                    num_microbatches:int,
                    backend:str)->None:
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="29502"
    
    dist.init_process_group(backend=backend,rank=rank,world_size=world_size)
    
    device=torch.device("cpu")
    
    stage=PipelineStage(hidden_size,hidden_size,rank).to(device)
    
    timings=[]
    for mb_idx in range(num_microbatches):
        start_time=time.perf_counter()
        
        if rank==0:
            activations=torch.randn(batch_size,hidden_size,device=device)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Created input "
                  f"(shape: {list(activations.shape)})")
        else :
            activations=torch.zeros(batch_size,hidden_size,device=device)
            dist.recv(activations,src=rank-1)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Received from stage {rank - 1}")
        
        with torch.no_grad():
            output=stage(activations)
        
        if rank ==world_size-1:
            print(f"[Stage {rank}] Microbatch {mb_idx}: Completed! "
                  f"Output mean: {output.mean().item():.4f}")
        else :
            dist.send(output,dst=rank+1)
            print(f"[Stage {rank}] Microbatch {mb_idx}: Sent to stage {rank + 1}")
        
        elapsed = time.perf_counter() - start_time
        timings.append(elapsed)
        
    dist.barrier()
    
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("PIPELINE SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Stages: {world_size}")
        print(f"Microbatches: {num_microbatches}")
        print(f"Batch size per microbatch: {batch_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"\nIn a real pipeline:")
        print(f"  - Stages process different microbatches in parallel")
        print(f"  - Backward pass sends gradients in reverse")
        print(f"  - 1F1B schedule optimizes memory usage")
        print(f"{'='*60}")
    
    dist.destroy_process_group()
    
    
    
    

def main():
    parser=argparse.ArgumentParser(description="Simulate a simple pipeline parallelism pattern with send/recv")
    parser.add_argument("--world-size","-w",
                        type=int,
                        default=4,
                        help="Number of pipeline stages (default : 4)")
    parser.add_argument("--backend",
                        default="gloo",
                        type=str,
                        choices=["gloo","nccl"],
                        help="Distributed backend")
    parser.add_argument("--batch-size","-b",
                           type=int,
                           default=32,
                           help="Batch Size for microbatch (default : 32)")
    parser.add_argument("--hidden-size",
                        type=int,
                        default=64,
                        help="Model hidden dimension")
    parser.add_argument("--num-microbatches","-m",
                        type=int,
                        default=4,
                        help="Number of microbatches to process (default: 4)")
    parser.add_argument("--visualize",
                        action="store_true",
                        help="Show pipeline visualization and exist")
    args=parser.parse_args()
    
    if args.visualize:
        visualize_pipeline()
        return

    print("=" * 60)
    print(" PIPELINE PARALLELISM SIMULATION")
    print("=" * 60)
    print(f"Number of stages: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Microbatches: {args.num_microbatches}")
    print("=" * 60 + "\n")
    
    mp.spawn(pipeline_worker,
             args=(args.world_size,args.batch_size,args.hidden_size,args.num_microbatches,args.backend),
             nprocs=args.world_size,
             join=True)

if __name__=='__main__':
    main()    