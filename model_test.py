from PIL import Image
from torch import nn
from torch.autograd import Variable
import torchvision
import torch.utils.data
from torchvision import transforms
import cv2


path = "1212.jpg"

dtype = torch.FloatTensor
loader = transforms.Compose([
    transforms.ToTensor()
])


def Image_open(path):
    image = Image.open(path)
    image = Variable(loader(image))
    image = image.unsqueeze(0)

    return image.type(dtype)


def resize(name):
    img = cv2.imread(name)
    resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(name, resized)


model = torchvision.models.resnet18()
model.eval()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("resnet18.pkl"))
resize(path)

image = Image_open(path)
outputs = model(image)
_, pre = torch.max(outputs.data, 1)
# print(torch.max(pre, 1))
classes = ('cat', 'stairs')

print('Predicted: ', ''.join('%5s' % classes[pre[0][0]]))
