"""
Setup Verification Script
=========================
Checks that all components are ready for the genomics Quixer pipeline.
"""

from pathlib import Path
import sys

def check_file(path, description):
    """Check if file exists."""
    path = Path(path)
    if path.exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {path}")
        return False

def check_dir(path, description):
    """Check if directory exists."""
    path = Path(path)
    if path.exists() and path.is_dir():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {path}")
        return False

def check_import(module_name):
    """Check if module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} installed")
        return True
    except ImportError:
        print(f"❌ {module_name} NOT INSTALLED")
        return False

def main():
    print("=" * 70)
    print("GENOMICS QUIXER SETUP VERIFICATION")
    print("=" * 70)
    
    all_good = True
    
    # Check data files
    print("\n📁 Data Files:")
    all_good &= check_dir(
        "/scratch/cbjp404/bradford_hackathon_2025/GRCh38_genomic_dataset",
        "GRCh38 dataset"
    )
    all_good &= check_file(
        "/scratch/cbjp404/bradford_hackathon_2025/GRCh38_genomic_dataset/GRCh38_latest_rna_summary.csv",
        "RNA summary CSV"
    )
    
    # Check scripts
    print("\n📝 Pipeline Scripts:")
    all_good &= check_file(
        "/scratch/cbjp404/bradford_hackathon_2025/preprocess_genomics.py",
        "Preprocessing script"
    )
    all_good &= check_file(
        "/scratch/cbjp404/bradford_hackathon_2025/lambeq_encoder.py",
        "lambeq encoder"
    )
    all_good &= check_file(
        "/scratch/cbjp404/bradford_hackathon_2025/run_genomics_training.py",
        "Training script"
    )
    all_good &= check_file(
        "/scratch/cbjp404/bradford_hackathon_2025/run_genomics_quixer.sh",
        "Slurm submission script"
    )
    
    # Check Quixer integration
    print("\n🔧 Quixer Integration:")
    all_good &= check_dir(
        "/scratch/cbjp404/Quixer",
        "Quixer directory"
    )
    all_good &= check_file(
        "/scratch/cbjp404/Quixer/quixer/setup_genomics.py",
        "Genomics data loader"
    )
    
    # Check Python dependencies
    print("\n📦 Python Dependencies:")
    all_good &= check_import("torch")
    all_good &= check_import("numpy")
    all_good &= check_import("pandas")
    all_good &= check_import("tqdm")
    all_good &= check_import("sklearn")
    
    lambeq_ok = check_import("lambeq")
    if not lambeq_ok:
        print("   ⚠️  lambeq not installed, will use fallback encoder")
    
    # Check output directories
    print("\n📂 Output Directories:")
    for dir_path, desc in [
        ("/scratch/cbjp404/bradford_hackathon_2025/processed_data", "Processed data"),
        ("/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings", "Embeddings"),
        ("/scratch/cbjp404/bradford_hackathon_2025/results", "Results"),
        ("/scratch/cbjp404/bradford_hackathon_2025/logs", "Logs"),
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        check_dir(dir_path, desc)
    
    # Check GPU availability
    print("\n🎮 GPU Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  No CUDA GPUs detected (will use CPU)")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    # Final status
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to run!")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Test preprocessing:")
        print("   python /scratch/cbjp404/bradford_hackathon_2025/preprocess_genomics.py")
        print("\n2. Submit full pipeline:")
        print("   cd /scratch/cbjp404/bradford_hackathon_2025")
        print("   sbatch run_genomics_quixer.sh")
        print("\n3. Monitor job:")
        print("   squeue -u $USER")
        print("   tail -f logs/quixer_*.out")
    else:
        print("❌ SETUP INCOMPLETE - Fix issues above before running")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
