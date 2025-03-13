import pandas as pd
from tqdm import tqdm
from src.modeling import ProteinTextLanguageModel
from datasets import load_dataset
from torchmetrics.text import EditDistance, TranslationEditRate

def main():
    model = ProteinTextLanguageModel.load_from_checkpoint('lightning_logs/version_1/checkpoints/last-v1.ckpt')
    model = model.to('cuda')
    
    edit_distance = EditDistance()
    ter = TranslationEditRate()
    
    dataset = load_dataset('khairi/ptlm-tiny-dataset')
    dataset = dataset['test']
    
    targets = []
    predictions = []
    
    for row in tqdm(dataset):
        sequence = row['Sequence']
        target = row['Answer'].replace("ASSISTANT:", "").strip()
        output = model.predict_step(sequence)
        
        targets.append(target.lower())
        predictions.append(output[0].lower())
        
    score = edit_distance(predictions, targets).item()
    score1 = ter(predictions, targets).item()
    
    pd.DataFrame({
        "prediction": predictions,
        "targets": targets
    }).to_csv('predictions.csv', index=False)
    
    print(f'Average edit distance: {score}')
    print(f'Average translation edit rate: {score1}')


if __name__ == '__main__':
    main()
