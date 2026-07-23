import torch

checkpoint = torch.load('./logs/dataset/G_114000.pth', map_location='cpu')

if 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# 包装成 api.py 期望的格式
torch.save({'model': state_dict}, 'checkpoint.pth')
