from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM

def main():
    dataset_id = "khairi/ptlm-dataset"
    protein_encoder_id = "khairi/Esm2-8M"
    language_model_id = "khairi/SmolLM-135M"

    AutoTokenizer.from_pretrained(protein_encoder_id)
    AutoTokenizer.from_pretrained(language_model_id)
    
    encoder = AutoModel.from_pretrained(protein_encoder_id)
    language_model = LlamaForCausalLM.from_pretrained(language_model_id)
    
    print("=" * 100)
    print(encoder)
    print("=" * 100)
    print(language_model)
    print("=" * 100)
    
    dataset = load_dataset(dataset_id)
    print(dataset)


if __name__ == '__main__':
    main()
