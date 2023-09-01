import tqdm
import torch
import pandas as pd

from utils import Dataset, Network

if __name__ == '__main__':
    test_dataset = Dataset('test_sub.xlsx', 'test')
    # test_dataset = Dataset('test_final.xlsx', 'test')
    
    model = Network(num_classes=2)
    
    model.cuda()
    model.eval()

    model.load_state_dict(torch.load('./cnn.pt'))
    
    data = []

    for path, image in tqdm.tqdm(test_dataset, desc='Inference'):
        with torch.no_grad():
            logits = model(image.cuda().unsqueeze(0))
            preds = torch.sigmoid(logits)

        flame, smoke = preds[0].cpu().detach().numpy()

        data.append([path, flame, smoke])

    pd.DataFrame(data).to_excel('results.xlsx', header=['Name', 'Flame', 'Smoke'], index=False, sheet_name='main')