import logging
from main import run_pipeline
from module import LepauteConfig, load_data, TransformerModel, EquivariantDataset, train_industrial_loop

def execute_training_validation():
    logging.basicConfig(level=logging.INFO)
    print("Initiating full Structural Telemetry collection (CI/CD Mock Mode Activated)...")
    
    config = LepauteConfig(device="cpu")
    collected = run_pipeline(
        config=config,
        display_mode="json", 
        unlimited=False, 
        save_json=True,
        mock=True
    )
    
    if not collected:
        print("\n[CRITICAL] System exited without telemetry.")
        return

    data = load_data(config=config)
    print(f"\nSuccessfully verified {len(data)} continuous frame transitions.")
    
    dataset = EquivariantDataset(data_list=data, config=config)
    print(f"\nConstructing Multi-Task Contrastive Optimization. Valid Pairs: {len(dataset)}")
    
    model = TransformerModel(config=config).to(torch.device(config.device))
    final_loss = train_industrial_loop(model, dataset, config=config, epochs=5)
    
    print(f"[SUCCESS] Spatial Contrastive Convergence complete. Loss: {final_loss:.4f}")

if __name__ == "__main__":
    execute_training_validation()