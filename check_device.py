import torch

def main():
    print("=== PyTorch CUDA diagnostics ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA devices count: {num_devices}")
        for i in range(num_devices):
            print(f"--- Device {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"Memory total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA НЕ доступна")

    print("\n=== torch.backends ===")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

if __name__ == "__main__":
    main()
