

import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torch



image=Image.open("./datasets/Capitals64/BASE/Code New Roman.0.0.png")
image = image.convert("L")
image=np.asarray(image)
image=image.reshape((-1,64,1664))
image=image/255

image = torch.from_numpy(image)

transform = transforms.Compose([
    transforms.Resize((64)),
    transforms.Normalize(( 0.5),
                        ( 0.5)),
    
    ])

img_tensor = transform(image)
img_tensor1=torch.reshape(img_tensor,(1,26,64,64))
img_tensor2=torch.tile(img_tensor1,(120,1,1,1))
print(img_tensor2.size())