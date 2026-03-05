import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def chain_worker(rank:int,world_size:int,backend:str)->None:
    
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="29501"
    
    dist.init_process_group(backend=backend,rank=rank,world_size=world_size)
    
    device=torch.device("cpu")
    if backend=="nccl" and torch.cuda.is_available():
        local_rank=rank%torch.cuda.device_count
        device=torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    if rank==0:
        tensor=torch.tensor([42.0],device=device)
        print(f"[Rank 0] starting chain with value: {tensor.item()}")
        dist.send(tensor,dst=rank+1)
        print(f"[Rank 0] sent to rank {rank+1}")
    elif rank==world_size-1:
        tensor=torch.zeros(1,device=device)
        dist.recv(tensor,src=rank-1)
        print(f"[Rank {rank}] Received final value: {tensor.item()}")
        print(f"\n{'='*50}")
        print(f"Chain complete!")
        print(f"Original: 42.0")
        print(f"After {world_size - 1} additions of 10: {tensor.item()}")
        print(f"Expected: {42.0 + (world_size - 1) * 10}")
        print(f"{'='*50}")
    else:
        tensor=torch.zeros(1,device=device)
        dist.recv(tensor,src=rank-1)
        print(f"Rank {rank} received tensor with value {tensor.item()} from rank {rank-1}")
        
        tensor+=10
        print(f"[Rank {rank}] after adding 10: {tensor.item()}")
        
        dist.send(tensor,dst=rank+1)
        print(f"[Rank {rank}] sent to rank {rank+1}")
        
    dist.barrier()
    dist.destroy_process_group()

def demonstrate_deadlock_pattern():
    """
    Educational function showing a deadlock pattern (DO NOT RUN).
    """
    print("""
    ⚠️  DEADLOCK PATTERN (DO NOT USE):

    # Process 0                # Process 1
    send(tensor, dst=1)        send(tensor, dst=0)
    recv(tensor, src=1)        recv(tensor, src=0)

    Both processes block on send(), waiting for the other to receive.
    Neither can proceed → DEADLOCK!

    ✓ CORRECT PATTERN (interleaved):

    # Process 0                # Process 1
    send(tensor, dst=1)        recv(tensor, src=0)
    recv(tensor, src=1)        send(tensor, dst=0)

    Process 0 sends while Process 1 receives → both can proceed.
    """)


def main():
    parser=argparse.ArgumentParser(description="Demonstrate chain pattern point-to-point Communication")
    parser.add_argument("--world-size","-w",
                        type=int,
                        default=4,
                        help="Number of processes to spawn (default: 4)")
    parser.add_argument("--backend",'-b',
                        type=str,
                        default="gloo",
                        choices=["gloo","nccl"],
                        help="Distributed backend (default: gloo for CPU compatibility)")
    parser.add_argument("--show-deadlock",'-d',
                        action='store_true',
                        help="Show deadlock pattern explanation"
                        )
    args=parser.parse_args()
    if args.show_deadlock:
        demonstrate_deadlock_pattern()
        
    mp.spawn(
        chain_worker,
        args=(args.world_size,args.backend),
        nprocs=args.world_size,
    )
    
    print("=" * 50)
    print(" POINT-TO-POINT COMMUNICATION: CHAIN PATTERN")
    print("=" * 50)
    print(f"World size: {args.world_size}")
    print(f"Pattern: Rank 0 → Rank 1 → ... → Rank {args.world_size - 1}")
    print(f"Operation: Each rank adds 10 before forwarding")
    print("=" * 50 + "\n")
    
    

    
if __name__=="__main__":
    main()